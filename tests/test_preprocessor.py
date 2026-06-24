"""Tests for video frame extraction and face preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

import core.preprocessor as preprocessor_module
from utils.video_utils import create_test_video


class DummyMTCNN:
    """Lightweight stand-in for facenet-pytorch's MTCNN."""

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        return np.array([[10.0, 10.0, 110.0, 110.0]]), np.array([0.99])


@pytest.fixture
def video_preprocessor(monkeypatch):
    """Create a preprocessor without loading the real MTCNN implementation."""
    monkeypatch.setattr(preprocessor_module, "MTCNN", DummyMTCNN)
    return preprocessor_module.VideoPreprocessor(device="cpu", frame_skip=5, max_frames=30)


def test_extract_frames_valid_video(tmp_path, video_preprocessor):
    """A generated short video should produce extracted frames."""
    video_path = create_test_video(tmp_path / "sample.mp4", seconds=5, fps=10)
    frames = video_preprocessor.extract_frames(str(video_path))
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)


def test_extract_frames_invalid_path(video_preprocessor):
    """Invalid paths should raise a ValueError."""
    with pytest.raises(ValueError):
        video_preprocessor.extract_frames("missing_video.mp4")


def test_face_detection_returns_dict(video_preprocessor):
    """Face detection should return the expected metadata structure."""
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
    results = video_preprocessor.detect_and_crop_faces(frames)
    assert len(results) == 3
    first_result = results[0]
    assert set(first_result.keys()) == {"frame_idx", "face", "bbox", "original", "tensors_by_size"}


def test_preprocess_face_uses_model_specific_input_size(video_preprocessor):
    """The preprocessor should honor the requested model-specific input size."""
    image = preprocessor_module.Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    tensor = video_preprocessor.preprocess_face(image, input_size=299)
    assert tuple(tensor.shape) == (1, 3, 299, 299)
