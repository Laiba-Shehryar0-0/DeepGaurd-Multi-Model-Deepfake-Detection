"""Central registry for DeepGuard's supported deepfake detectors."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HALF_RANGE_MEAN = [0.5, 0.5, 0.5]
HALF_RANGE_STD = [0.5, 0.5, 0.5]


def _model_entry(
    *,
    display_name: str,
    subtitle: str,
    auc: float,
    score_display: str,
    benchmark_label: str,
    input_size: int,
    weight_file: str,
    timm_name: str | None,
    gradcam_layer: str,
    normalize_mean: list[float],
    normalize_std: list[float],
    speed: str,
    description: str,
    recommended: bool,
    vit_reshape: bool = False,
    real_class_idx: int = 0,
    fake_class_idx: int = 1,
) -> dict:
    """Build one concise model-registry entry."""

    return {
        "display_name": display_name,
        "subtitle": subtitle,
        "auc": auc,
        "score_display": score_display,
        "benchmark_label": benchmark_label,
        "input_size": input_size,
        "weight_file": weight_file,
        "timm_name": timm_name,
        "gradcam_layer": gradcam_layer,
        "vit_reshape": vit_reshape,
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
        "real_class_idx": real_class_idx,
        "fake_class_idx": fake_class_idx,
        "speed": speed,
        "description": description,
        "recommended": recommended,
    }


MODEL_REGISTRY = {
    "xception": _model_entry(
        display_name="Xception (FaceForge)",
        subtitle="Best Public Single-Frame Detector",
        auc=0.9995,
        score_display="99.33% acc / AUC 0.9995",
        benchmark_label="99.33% acc / AUC 0.9995 on FaceForensics++",
        input_size=224,
        weight_file="models/xception_faceforge.pth",
        timm_name="xception",
        gradcam_layer="xception.conv4",
        normalize_mean=HALF_RANGE_MEAN,
        normalize_std=HALF_RANGE_STD,
        speed="Medium",
        description="Public FaceForge Xception checkpoint with the strongest published single-frame benchmark in this project.",
        recommended=True,
    ),
    "vit_base": _model_entry(
        display_name="ViT-B/16",
        subtitle="Public ViT Face Classifier",
        auc=0.9870,
        score_display="Accuracy 98.70%",
        benchmark_label="98.70% accuracy on test set",
        input_size=224,
        weight_file="models/vit_deepfake_wvolf.pth",
        timm_name="vit_base_patch16_224",
        gradcam_layer="vit.layers[-1].layernorm_before",
        normalize_mean=HALF_RANGE_MEAN,
        normalize_std=HALF_RANGE_STD,
        speed="Medium",
        description="Public single-image ViT checkpoint with strong deepfake classification performance.",
        recommended=True,
        vit_reshape=True,
    ),
    "resnet50_ffpp": _model_entry(
        display_name="ResNet-50 (FF++)",
        subtitle="FF++ C23 Public Baseline",
        auc=0.9450,
        score_display="94.87% acc / AUC 0.9450",
        benchmark_label="94.87% acc / AUC 0.945 on FF++",
        input_size=224,
        weight_file="models/resnet50_ffpp.pth",
        timm_name="resnet50",
        gradcam_layer="net[0].layer4[-1]",
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        speed="Fast",
        description="Public ResNet-50 baseline trained for FaceForensics++ style detection.",
        recommended=True,
    ),
    "efficientnet_b7": _model_entry(
        display_name="EfficientNet-B7",
        subtitle="High-Accuracy Optimized",
        auc=0.9280,
        score_display="AUC 0.9280",
        benchmark_label="AUC 0.928",
        input_size=224,
        weight_file="models/efficientnet_b7_deepfake.pth",
        timm_name="efficientnet_b7",
        gradcam_layer="blocks[-1]",
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        speed="Slow",
        description="Largest EfficientNet option for higher accuracy when runtime cost is acceptable.",
        recommended=True,
    ),
    "efficientnet_b4": _model_entry(
        display_name="EfficientNet-B4",
        subtitle="Balanced Speed and Accuracy",
        auc=0.9000,
        score_display="AUC 0.9000",
        benchmark_label="AUC 0.900",
        input_size=224,
        weight_file="models/efficientnet_b4_deepfake.pth",
        timm_name="efficientnet_b4",
        gradcam_layer="blocks[-1]",
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        speed="Fast",
        description="Balanced EfficientNet model for practical frame-level deepfake detection.",
        recommended=True,
    ),
    "mesonet4": _model_entry(
        display_name="MesoNet-4",
        subtitle="Lightweight Baseline",
        auc=0.8200,
        score_display="~98% acc (limited sets)",
        benchmark_label="Lightweight comparison model",
        input_size=256,
        weight_file="models/mesonet4.pth",
        timm_name=None,
        gradcam_layer="conv4",
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        speed="Very Fast",
        description="Small CNN baseline included mainly for comparison and low-resource runs.",
        recommended=False,
    ),
}


def get_available_models() -> dict:
    """Return only models whose weight files exist on disk and are non-empty."""

    available: dict[str, dict] = {}
    for model_key in MODEL_REGISTRY:
        config = get_model_config(model_key)
        if config["weight_path"].exists() and config["weight_path"].stat().st_size > 1000:
            available[model_key] = config
    return available


def get_model_config(model_key: str) -> dict:
    """Return a registry config copy for one model key."""

    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_key}' not in registry. Valid keys: {list(MODEL_REGISTRY.keys())}")
    config = deepcopy(MODEL_REGISTRY[model_key])
    config["model_key"] = model_key
    config["weight_path"] = ROOT_DIR / config["weight_file"]
    return config
