"""Visualization helpers for GradCAM and Streamlit display."""

from __future__ import annotations

import numpy as np


def build_side_by_side(original_rgb: np.ndarray, heatmap_overlay_rgb: np.ndarray) -> np.ndarray:
    """Concatenate the original face and GradCAM overlay horizontally."""
    return np.concatenate([original_rgb, heatmap_overlay_rgb], axis=1)


def list_suspicious_regions(gradcam_results: list[dict]) -> str:
    """Create a readable summary of unique suspicious regions."""
    regions = []
    for result in gradcam_results:
        region = result.get("suspicious_region")
        if region and region not in regions:
            regions.append(region)
    return ", ".join(regions) if regions else "N/A"
