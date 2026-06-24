"""GradCAM-based explanation utilities for DeepGuard."""

from __future__ import annotations

import logging
import re

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
from torchvision import transforms


LOGGER = logging.getLogger("deepguard.explainer")
TOKEN_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)|\[(-?\d+)\]")


class _GradCAMLogitsWrapper(torch.nn.Module):
    """Wrap model outputs so GradCAM always receives a tensor of logits."""

    def __init__(self, model: torch.nn.Module):
        """Store the original model as a submodule."""
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """Return raw logits for GradCAM regardless of model backend."""
        outputs = self.model(*args, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs


class GradCAMExplainer:
    """Generate GradCAM explanations for the winning DeepGuard model."""

    def __init__(self, detector, device: str):
        """
        Initialize GradCAM for the winning detector model.

        Args:
            detector: Loaded `DeepfakeDetector` instance chosen by the ensemble resolver.
            device: Device string used for inference.
        """
        self.detector = detector
        self.model = detector.model
        self.cam_model = _GradCAMLogitsWrapper(self.model) if detector.backend == "transformers" else self.model
        self.device = device
        self.input_size = detector.input_size
        self.normalize_mean = detector.normalize_mean
        self.normalize_std = detector.normalize_std
        self.preprocess_cache_key = detector.preprocess_cache_key
        self.fake_class_idx = detector.fake_class_idx
        self.target_layer = self._resolve_target_layer()
        self.reshape_transform = vit_reshape_transform if detector.config["vit_reshape"] else None

    def _resolve_path(self, root, path: str):
        """Resolve a registry layer path such as `blocks[-1].norm1`."""
        current = root
        for segment in path.split("."):
            tokens = TOKEN_PATTERN.findall(segment)
            for attribute_name, index_token in tokens:
                if attribute_name:
                    current = getattr(current, attribute_name)
                elif index_token:
                    current = current[int(index_token)]
        return current

    def _resolve_target_layer(self):
        """Resolve the configured GradCAM target layer or fall back sensibly."""
        configured_path = self.detector.config["gradcam_layer"]
        try:
            target_layer = self._resolve_path(self.model, configured_path)
            LOGGER.info("Using configured GradCAM layer `%s` for `%s`.", configured_path, self.detector.model_key)
            return target_layer
        except Exception:
            LOGGER.warning(
                "Configured GradCAM layer `%s` not found for `%s`. Falling back to auto-detection.",
                configured_path,
                self.detector.model_key,
            )

        named_modules = list(self.model.named_modules())
        if self.detector.model_key == "vit_base":
            for name, module in reversed(named_modules):
                if isinstance(module, torch.nn.LayerNorm):
                    LOGGER.info("Auto-detected ViT GradCAM layer: %s", name)
                    return module

        for name, module in reversed(named_modules):
            if isinstance(module, torch.nn.Conv2d):
                LOGGER.info("Auto-detected convolutional GradCAM layer: %s", name)
                return module

        for name, module in reversed(named_modules):
            if isinstance(module, torch.nn.Module):
                LOGGER.info("Using final module `%s` as GradCAM fallback.", name)
                return module
        raise ValueError("Unable to locate a GradCAM target layer.")

    def _build_input_tensor(self, face_image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess an RGB face crop to the winning model's input size."""
        transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(self.normalize_mean),
                    std=list(self.normalize_std),
                ),
            ]
        )
        image = Image.fromarray(face_image_rgb.astype(np.uint8))
        return transform(image).unsqueeze(0).to(self.device)

    def generate_heatmap(self, face_tensor: torch.Tensor, face_image_rgb: np.ndarray) -> dict:
        """
        Generate a GradCAM heatmap overlay for the FAKE class.

        Args:
            face_tensor: Preprocessed tensor aligned with the winning model's input size.
            face_image_rgb: Original RGB face crop for overlay rendering.

        Returns:
            Dictionary with overlay image, raw activation map, and suspicious region label.
        """
        input_tensor = face_tensor.to(self.device) if face_tensor is not None else self._build_input_tensor(face_image_rgb)
        resized_face = np.array(Image.fromarray(face_image_rgb.astype(np.uint8)).resize((self.input_size, self.input_size)))
        normalized_face = resized_face.astype(np.float32)
        if normalized_face.max() > 1.0:
            normalized_face = normalized_face / 255.0

        with GradCAM(
            model=self.cam_model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        ) as cam:
            raw_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(self.fake_class_idx)])[0]

        heatmap_overlay = show_cam_on_image(normalized_face, raw_cam, use_rgb=True)
        suspicious_region = self.identify_suspicious_region(raw_cam)
        return {
            "heatmap_overlay": heatmap_overlay,
            "raw_cam": raw_cam,
            "suspicious_region": suspicious_region,
            "resized_face_rgb": resized_face,
        }

    def identify_suspicious_region(self, raw_cam: np.ndarray) -> str:
        """
        Divide the face into a 3x3 grid and label the highest-activation region.

        Args:
            raw_cam: Grayscale activation map from GradCAM.

        Returns:
            Human-readable region label.
        """
        labels = [
            "forehead area",
            "upper left face",
            "upper right face",
            "left eye region",
            "nose bridge",
            "right eye region",
            "left jaw",
            "chin area",
            "right jaw",
        ]
        grid_rows = np.array_split(raw_cam, 3, axis=0)
        region_means: list[float] = []
        for row in grid_rows:
            for region in np.array_split(row, 3, axis=1):
                region_means.append(float(np.mean(region)))
        return labels[int(np.argmax(region_means))]

    def process_top_frames(self, top_frame_indices: list, face_data: list) -> list[dict]:
        """
        Generate GradCAM explanations for the winning model's top suspicious frames.

        Args:
            top_frame_indices: Original frame indices selected by the winning model's aggregator.
            face_data: Face metadata records returned by the preprocessor.

        Returns:
            List of GradCAM result dictionaries for up to three frames.
        """
        face_lookup = {item["frame_idx"]: item for item in face_data}
        results: list[dict] = []

        for frame_idx in top_frame_indices[:3]:
            item = face_lookup.get(frame_idx)
            if item is None:
                continue

            face_image_rgb = np.array(item["face"].convert("RGB"))
            tensors_by_size = item.get("tensors_by_size", {})
            face_tensor = tensors_by_size.get(self.preprocess_cache_key)
            heatmap = self.generate_heatmap(face_tensor=face_tensor, face_image_rgb=face_image_rgb)
            heatmap["frame_idx"] = item["frame_idx"]
            heatmap["face_image_rgb"] = face_image_rgb
            heatmap["model_key"] = self.detector.model_key
            results.append(heatmap)

        return results
