"""Tests for the DeepGuard frame-level detector."""

from __future__ import annotations

import torch

from core.detector import DeepfakeDetector


class DummyModel(torch.nn.Module):
    """Simple fixed-logit model for deterministic detector tests."""

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        logits = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


def make_detector() -> DeepfakeDetector:
    """Construct a detector instance without loading files from disk."""
    detector = object.__new__(DeepfakeDetector)
    detector.model_key = "efficientnet_b4"
    detector.config = {"input_size": 224, "display_name": "EfficientNet-B4", "auc": 0.9}
    detector.device = "cpu"
    detector.model_path = None
    detector.runtime_warnings = []
    detector.weight_source = "test"
    detector.backend = "timm"
    detector.is_generic_fallback = False
    detector.needs_fine_tuning = False
    detector.model = DummyModel()
    detector.model.eval()
    return detector


def test_predict_frame_output_structure():
    """The frame prediction output should contain all required detector keys."""
    detector = make_detector()
    result = detector.predict_frame(torch.randn(1, 3, 224, 224))
    assert {"label", "fake_prob", "real_prob", "logits", "model_key"}.issubset(result.keys())


def test_confidence_between_zero_and_one():
    """Fake probability scores should be normalized probabilities."""
    detector = make_detector()
    result = detector.predict_frame(torch.randn(1, 3, 224, 224))
    assert 0.0 <= result["fake_prob"] <= 1.0


def test_label_is_real_or_fake():
    """Predicted labels should always be REAL or FAKE."""
    detector = make_detector()
    result = detector.predict_frame(torch.randn(1, 3, 224, 224))
    assert result["label"] in {"REAL", "FAKE"}
