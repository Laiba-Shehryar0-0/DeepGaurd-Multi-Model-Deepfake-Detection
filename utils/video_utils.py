"""Video and image helpers used across the DeepGuard project."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger("deepguard.video_utils")


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """Open a video file using a few OpenCV backends before failing."""
    candidate_backends = [
        getattr(cv2, "CAP_FFMPEG", None),
        getattr(cv2, "CAP_AVFOUNDATION", None),
        getattr(cv2, "CAP_ANY", None),
    ]

    for backend in candidate_backends:
        if backend is None:
            continue
        capture = cv2.VideoCapture(str(video_path), backend)
        if capture.isOpened():
            return capture
        capture.release()

    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        return capture

    raise ValueError(f"Video not readable: {video_path}")


def frame_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array into uint8 for display or file output."""
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> None:
    """Persist an RGB image using OpenCV."""
    output = ensure_uint8(image)
    cv2.imwrite(str(output_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def create_test_video(
    video_path: str | Path,
    seconds: int = 5,
    fps: int = 10,
    size: tuple[int, int] = (320, 240),
) -> Path:
    """Generate a small deterministic test video for automated tests."""
    video_path = Path(video_path)
    fourcc_candidates = ("mp4v", "avc1", "XVID")
    writer = None
    for fourcc_name in fourcc_candidates:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            size,
        )
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None or not writer.isOpened():
        raise RuntimeError("Unable to create a test video with the available codecs.")

    total_frames = seconds * fps
    width, height = size
    for frame_index in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (frame_index * 5) % 255
        frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)
        frame[:, :, 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
        cv2.putText(
            frame,
            f"F{frame_index:02d}",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    LOGGER.info("Created test video at %s", video_path)
    return video_path
