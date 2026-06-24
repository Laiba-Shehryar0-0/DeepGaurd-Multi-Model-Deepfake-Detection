"""Single-model inference wrapper for DeepGuard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from core.model_registry import get_model_config

try:
    from transformers import AutoConfig, AutoModelForImageClassification
except Exception:  # pragma: no cover - import failure handled by environment checks
    AutoConfig = None
    AutoModelForImageClassification = None


LOGGER = logging.getLogger("deepguard.detector")


class MesoNet4(nn.Module):
    """PyTorch implementation of the 4-layer MesoNet architecture."""

    def __init__(self):
        """Create the lightweight MesoNet-4 classifier."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Run MesoNet-4 forward inference."""
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class FFPPResNet50(nn.Module):
    """ResNet-50 backbone plus the lightweight binary head used by the public FF++ checkpoint."""

    def __init__(self, pretrained_backbone: bool = False):
        """Create the FF++ ResNet-50 architecture expected by the public PyDeepFakeDet weights."""
        super().__init__()
        backbone = timm.create_model("resnet50", pretrained=pretrained_backbone, num_classes=1000)
        self.net = nn.Sequential(backbone, nn.Linear(1000, 2))

    def forward(self, x):
        """Run forward inference through the wrapped FF++ ResNet-50 model."""
        return self.net(x)


class FaceForgeXception(nn.Module):
    """Xception backbone plus the custom FaceForge binary head."""

    def __init__(self, pretrained_backbone: bool = False):
        """Create the architecture expected by the public FaceForge checkpoint."""
        super().__init__()
        self.xception = timm.create_model("xception", pretrained=pretrained_backbone, num_classes=0, global_pool="avg")
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.xception.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        """Run forward inference through the FaceForge Xception classifier."""
        features = self.xception(x)
        return self.classifier(features)


class DeepfakeDetector:
    """Load a single registry-backed detector and run frame-level predictions."""

    def __init__(self, model_key: str, device: str):
        """
        Load the model identified by `model_key` from the model registry.

        Args:
            model_key: Registry key for the requested model.
            device: Preferred device string.
        """
        self.model_key = model_key
        self.config = get_model_config(model_key)
        self.model_path = Path(self.config["weight_path"])
        self.device = self._resolve_device(device)
        self.runtime_warnings: list[str] = []
        self.weight_source = "unknown"
        self.backend = "timm"
        self.is_generic_fallback = False
        self.needs_fine_tuning = False
        self.model = self._build_model()
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    @property
    def input_size(self) -> int:
        """Return the required input size for this detector from the registry."""
        return int(self.config["input_size"])

    @property
    def normalize_mean(self) -> tuple[float, float, float]:
        """Return the normalization mean required by the active detector."""
        values = self.config.get("normalize_mean", [0.485, 0.456, 0.406])
        return tuple(float(value) for value in values)

    @property
    def normalize_std(self) -> tuple[float, float, float]:
        """Return the normalization standard deviation required by the active detector."""
        values = self.config.get("normalize_std", [0.229, 0.224, 0.225])
        return tuple(float(value) for value in values)

    @property
    def real_class_idx(self) -> int:
        """Return the index representing the REAL class."""
        return int(self.config.get("real_class_idx", 0))

    @property
    def fake_class_idx(self) -> int:
        """Return the index representing the FAKE class."""
        return int(self.config.get("fake_class_idx", 1))

    @property
    def preprocess_cache_key(self) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
        """Return a cache key that includes both size and normalization settings."""
        return (self.input_size, self.normalize_mean, self.normalize_std)

    @property
    def auc(self) -> float:
        """Return the published benchmark score used as this model's ensemble weight."""
        return float(self.config["auc"])

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve the effective device for inference."""
        if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for %s but unavailable. Falling back to CPU.", self.model_key)
            return "cpu"
        return "cuda" if requested_device.lower().startswith("cuda") else "cpu"

    def _fallback_to_cpu(self, reason: str) -> None:
        """Move the active model to CPU after an out-of-memory condition."""
        if self.device == "cpu":
            return
        warning = f"{reason} Falling back to CPU for classification."
        LOGGER.warning(warning)
        self.runtime_warnings.append(warning)
        self.device = "cpu"
        self.model.to(self.device)

    def _build_model(self) -> nn.Module:
        """Build the architecture associated with the current registry entry."""
        if self.model_key == "mesonet4":
            return MesoNet4()
        if self.model_key == "resnet50_ffpp":
            return FFPPResNet50(pretrained_backbone=False)
        if self.model_key == "xception":
            return FaceForgeXception(pretrained_backbone=False)
        if self.model_key == "vit_base":
            return timm.create_model(self.config["timm_name"], pretrained=False, num_classes=2)
        return timm.create_model(self.config["timm_name"], pretrained=False, num_classes=2)

    def _extract_state_dict(self, payload: Any) -> dict[str, torch.Tensor]:
        """Normalize checkpoint payloads to a raw state dictionary."""
        if isinstance(payload, dict):
            self.weight_source = str(payload.get("source", self.weight_source))
            self.backend = str(payload.get("backend", self.backend))
            self.is_generic_fallback = bool(payload.get("is_generic_fallback", False))
            self.needs_fine_tuning = bool(payload.get("needs_fine_tuning", False))
            for key in ("state_dict", "model_state_dict", "model_state", "model", "net"):
                maybe_state = payload.get(key)
                if isinstance(maybe_state, dict):
                    return self._sanitize_state_dict(maybe_state)
            if all(isinstance(value, torch.Tensor) for value in payload.values()):
                return self._sanitize_state_dict(payload)
        raise ValueError("Unsupported checkpoint format.")

    def _sanitize_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Strip common wrapping prefixes from saved parameter keys."""
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "model.", "encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value
        return cleaned

    def _build_transformers_model(self, payload: dict[str, Any]) -> nn.Module:
        """Recreate a transformers ViT classifier from serialized config metadata."""
        if AutoConfig is None or AutoModelForImageClassification is None:
            raise ImportError("transformers is required to load the ViT backend.")

        config_dict = payload.get("config")
        if not isinstance(config_dict, dict):
            raise ValueError("Transformers payload missing `config` metadata.")
        config_payload = dict(config_dict)
        model_type = config_payload.pop("model_type")
        config = AutoConfig.for_model(model_type, **config_payload)
        model = AutoModelForImageClassification.from_config(config)
        self.backend = "transformers"
        return model

    def _load_weights(self) -> None:
        """Load the checkpoint from disk using non-strict state-dict matching."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. Run `python models/download_all.py` first."
            )

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("build_name") and payload.get("build_name") != self.config.get("timm_name"):
            self.model = timm.create_model(payload["build_name"], pretrained=False, num_classes=2)
        if isinstance(payload, dict) and payload.get("backend") == "transformers":
            self.model = self._build_transformers_model(payload)

        state_dict = self._extract_state_dict(payload)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            LOGGER.warning(
                "Checkpoint loaded with key mismatch for %s. missing=%s unexpected=%s",
                self.model_key,
                missing_keys,
                unexpected_keys,
            )
        LOGGER.info("Loaded detector `%s` from %s", self.model_key, self.model_path)

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with automatic CPU fallback on OOM."""
        try:
            outputs = self.model(input_tensor.to(self.device))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                self._fallback_to_cpu(f"CUDA out of memory during {self.model_key} inference.")
                outputs = self.model(input_tensor.to(self.device))
            else:
                raise

        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs

    def _logits_to_result(self, logits: torch.Tensor) -> dict:
        """Convert raw logits into a structured prediction dictionary."""
        detached_logits = logits.detach().cpu()

        if detached_logits.ndim == 1:
            detached_logits = detached_logits.unsqueeze(0)

        if detached_logits.shape[-1] == 1:
            fake_prob = float(torch.sigmoid(detached_logits[0, 0]).item())
            real_prob = 1.0 - fake_prob
        else:
            probabilities = torch.softmax(detached_logits, dim=1)[0]
            real_prob = float(probabilities[self.real_class_idx].item())
            fake_prob = float(probabilities[self.fake_class_idx].item())

        label = "FAKE" if fake_prob >= real_prob else "REAL"
        confidence = fake_prob if label == "FAKE" else real_prob
        return {
            "label": label,
            "confidence": float(confidence),
            "fake_prob": float(fake_prob),
            "real_prob": float(real_prob),
            "logits": detached_logits.squeeze(0),
            "model_key": self.model_key,
        }

    def predict_frame(self, face_tensor: torch.Tensor) -> dict:
        """
        Predict whether a single face crop is real or fake for this model key.

        Args:
            face_tensor: Tensor of shape `(1, 3, input_size, input_size)`.

        Returns:
            Prediction dictionary with label, probabilities, confidence, logits, and model_key.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(face_tensor)
        return self._logits_to_result(logits)

    def predict_batch(self, face_tensors: list) -> list[dict]:
        """
        Predict frame labels for a batch of face tensors for this model key.

        Args:
            face_tensors: List of normalized face tensors matching this detector's input_size.

        Returns:
            List of per-frame prediction dictionaries.
        """
        if not face_tensors:
            return []

        batch = torch.cat(face_tensors, dim=0)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(batch)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        return [self._logits_to_result(sample_logits.unsqueeze(0)) for sample_logits in logits]
