"""Tests for the DeepGuard model registry."""

from __future__ import annotations

from core.model_registry import get_model_config


def test_resnet50_ffpp_registry_entry():
    """The FF++ ResNet-50 model should keep its core runtime metadata."""
    config = get_model_config("resnet50_ffpp")
    assert config["display_name"] == "ResNet-50 (FF++)"
    assert config["auc"] == 0.945
    assert config["weight_file"] == "models/resnet50_ffpp.pth"
    assert config["timm_name"] == "resnet50"


def test_xception_registry_points_to_faceforge():
    """The Xception entry should keep the correct checkpoint and preprocessing info."""
    config = get_model_config("xception")
    assert config["weight_file"] == "models/xception_faceforge.pth"
    assert config["score_display"] == "99.33% acc / AUC 0.9995"
    assert config["normalize_mean"] == [0.5, 0.5, 0.5]


def test_vit_registry_points_to_real_vit_checkpoint():
    """The ViT entry should keep the correct checkpoint and ViT-specific flags."""
    config = get_model_config("vit_base")
    assert config["weight_file"] == "models/vit_deepfake_wvolf.pth"
    assert config["timm_name"] == "vit_base_patch16_224"
    assert config["vit_reshape"] is True
