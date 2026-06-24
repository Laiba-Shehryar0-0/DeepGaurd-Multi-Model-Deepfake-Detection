"""Frame extraction and face preprocessing for DeepGuard."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from utils.video_utils import frame_bgr_to_rgb, open_video_capture


LOGGER = logging.getLogger("deepguard.preprocessor")


class VideoPreprocessor:
    """Handle video frame extraction and MTCNN-based face cropping."""

    def __init__(self, device: str = "cpu", frame_skip: int = 10, max_frames: int = 30):
        """
        Create the shared video preprocessor.

        Args:
            device: Preferred inference device, either `cpu` or `cuda`.
            frame_skip: Extract every Nth frame.
            max_frames: Maximum number of frames to keep from the video.
        """
        self.device = self._resolve_device(device)
        self.frame_skip = max(1, frame_skip)
        self.max_frames = max(1, max_frames)
        self.runtime_warnings: list[str] = []
        self.last_face_count = 0
        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            device=self.device,
        )

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve the active device, falling back to CPU if CUDA is unavailable."""
        if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return "cuda" if requested_device.lower().startswith("cuda") else "cpu"

    def _fallback_to_cpu(self, reason: str) -> None:
        """Move MTCNN to CPU after an out-of-memory condition."""
        if self.device == "cpu":
            return
        warning = f"{reason} Falling back to CPU for face detection."
        LOGGER.warning(warning)
        self.runtime_warnings.append(warning)
        self.device = "cpu"
        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            device=self.device,
        )

    def extract_frames(self, video_path: str) -> list[np.ndarray]:
        """
        Extract frames from a video at the configured interval.

        Args:
            video_path: Readable path to a video file.

        Returns:
            A list of BGR numpy arrays.

        Raises:
            ValueError: If the video cannot be opened or yields no frames.
        """
        if not Path(video_path).exists():
            raise ValueError(f"Video not readable: {video_path}")

        capture = open_video_capture(video_path)
        frames: list[np.ndarray] = []
        frame_index = 0

        while len(frames) < self.max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % self.frame_skip == 0:
                frames.append(frame.copy())
            frame_index += 1

        capture.release()

        if not frames:
            raise ValueError(f"Video not readable: {video_path}")

        LOGGER.info("Extracted %s frames from %s", len(frames), video_path)
        return frames

    def _crop_face(
        self,
        frame_rgb: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Crop a face region from an RGB frame with padding."""
        height, width = frame_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        margin = 20
        left = max(0, int(round(x1 - margin)))
        top = max(0, int(round(y1 - margin)))
        right = min(width, int(round(x2 + margin)))
        bottom = min(height, int(round(y2 + margin)))
        face_crop = frame_rgb[top:bottom, left:right]
        return Image.fromarray(face_crop), (left, top, right, bottom)

    def detect_and_crop_faces(self, frames: list) -> list[dict]:
        """
        Detect the dominant face in each extracted frame.

        Args:
            frames: Sequence of BGR frames from OpenCV.

        Returns:
            List of dictionaries with `frame_idx`, `face`, `bbox`, and `original`.

        Raises:
            ValueError: `insufficient_faces` when fewer than 3 faces are found.
        """
        face_records: list[dict] = []

        for frame_idx, frame in enumerate(frames):
            frame_rgb = frame_bgr_to_rgb(frame)
            pil_frame = Image.fromarray(frame_rgb)

            try:
                boxes, probs = self.mtcnn.detect(pil_frame)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    self._fallback_to_cpu("CUDA out of memory during MTCNN inference.")
                    boxes, probs = self.mtcnn.detect(pil_frame)
                else:
                    raise

            if boxes is None or len(boxes) == 0:
                LOGGER.debug("No face detected in extracted frame %s", frame_idx)
                continue

            best_index = int(np.argmax(probs)) if probs is not None else 0
            best_bbox = tuple(float(value) for value in boxes[best_index])
            face_image, clipped_bbox = self._crop_face(frame_rgb, best_bbox)
            face_records.append(
                {
                    "frame_idx": frame_idx,
                    "face": face_image,
                    "bbox": clipped_bbox,
                    "original": frame.copy(),
                    "tensors_by_size": {},
                }
            )

        self.last_face_count = len(face_records)
        LOGGER.info("Detected %s face crops across %s frames", self.last_face_count, len(frames))

        if self.last_face_count < 3:
            raise ValueError("insufficient_faces")

        return face_records

    def preprocess_face(
        self,
        face_image: Image.Image,
        input_size: int = 224,
        mean: list[float] | tuple[float, float, float] | None = None,
        std: list[float] | tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """
        Convert a cropped face image to a normalized tensor.

        Args:
            face_image: PIL face crop in RGB color space.
            input_size: Required square resolution for the active model.
            mean: Per-channel normalization mean for the active model.
            std: Per-channel normalization standard deviation for the active model.

        Returns:
            A tensor of shape `(1, 3, input_size, input_size)`.
        """
        normalize_mean = list(mean or [0.485, 0.456, 0.406])
        normalize_std = list(std or [0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                ),
            ]
        )
        return transform(face_image).unsqueeze(0)
