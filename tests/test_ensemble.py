"""Tests for the DeepGuard AUC-based multi-model conflict resolver."""

from __future__ import annotations

import pytest

from core.ensemble import DeepfakeEnsemble


def make_results() -> dict:
    """Return a reusable mock per-model result mapping."""
    return {
        "xception": {
            "verdict": "FAKE",
            "video_fake_prob": 0.72,
            "fake_frame_count": 17,
            "real_frame_count": 3,
            "total_frames": 20,
            "top_suspicious_frames": [1, 3, 8],
            "fake_prob_timeline": [0.94, 0.91],
        },
        "vit_base": {
            "verdict": "FAKE",
            "video_fake_prob": 0.68,
            "fake_frame_count": 16,
            "real_frame_count": 4,
            "total_frames": 20,
            "top_suspicious_frames": [0, 2, 5],
            "fake_prob_timeline": [0.90, 0.88],
        },
        "efficientnet_b4": {
            "verdict": "REAL",
            "video_fake_prob": 0.31,
            "fake_frame_count": 8,
            "real_frame_count": 12,
            "total_frames": 20,
            "top_suspicious_frames": [4, 7, 9],
            "fake_prob_timeline": [0.22, 0.31],
        },
        "mesonet4": {
            "verdict": "FAKE",
            "video_fake_prob": 0.95,
            "fake_frame_count": 18,
            "real_frame_count": 2,
            "total_frames": 20,
            "top_suspicious_frames": [1, 2, 3],
            "fake_prob_timeline": [0.95, 0.95],
        },
    }


def make_ensemble(use_weighted_voting: bool = False) -> DeepfakeEnsemble:
    """Create an uninitialized ensemble shell for direct resolver tests."""
    ensemble = object.__new__(DeepfakeEnsemble)
    ensemble.use_weighted_voting = use_weighted_voting
    return ensemble


def test_single_model_resolution():
    """One model should resolve with `single` semantics."""
    ensemble = make_ensemble()
    resolved = ensemble._resolve({"xception": make_results()["xception"]})
    assert resolved["resolution_method"] == "single"
    assert resolved["agreement"] is True


def test_unanimous_all_fake():
    """Unanimous FAKE votes should choose the highest-AUC model."""
    ensemble = make_ensemble()
    results = make_results()
    results["efficientnet_b4"]["verdict"] = "FAKE"
    results["efficientnet_b4"]["video_fake_prob"] = 0.66
    resolved = ensemble._resolve({key: results[key] for key in ("xception", "vit_base", "efficientnet_b4")})
    assert resolved["resolution_method"] == "unanimous"
    assert resolved["winning_model"] == "xception"


def test_majority_vote_fake_wins():
    """A 2 FAKE vs 1 REAL split should resolve to FAKE with majority_vote."""
    ensemble = make_ensemble()
    resolved = ensemble._resolve({key: make_results()[key] for key in ("xception", "vit_base", "efficientnet_b4")})
    assert resolved["final_verdict"] == "FAKE"
    assert resolved["resolution_method"] == "majority_vote"
    assert resolved["winning_model"] == "xception"


def test_majority_vote_winner_is_auc_not_confidence():
    """Higher AUC must beat higher confidence among majority-side voters."""
    ensemble = make_ensemble()
    resolved = ensemble._resolve({key: make_results()[key] for key in ("xception", "mesonet4", "efficientnet_b4")})
    assert resolved["winning_model"] == "xception"
    assert resolved["final_verdict"] == "FAKE"


def test_tie_broken_by_auc():
    """A perfect tie should be broken by the highest-AUC model overall."""
    ensemble = make_ensemble()
    resolved = ensemble._resolve({key: make_results()[key] for key in ("xception", "efficientnet_b4")})
    assert resolved["resolution_method"] == "tie_broken_by_auc"
    assert resolved["winning_model"] == "xception"


def test_weighted_vote_formula():
    """The weighted soft-vote result should match the documented AUC formula."""
    ensemble = make_ensemble(use_weighted_voting=True)
    results = {
        "xception": {"verdict": "FAKE", "video_fake_prob": 0.8, "fake_frame_count": 1, "real_frame_count": 0, "total_frames": 1, "top_suspicious_frames": [], "fake_prob_timeline": [0.8]},
        "vit_base": {"verdict": "REAL", "video_fake_prob": 0.3, "fake_frame_count": 0, "real_frame_count": 1, "total_frames": 1, "top_suspicious_frames": [], "fake_prob_timeline": [0.3]},
        "efficientnet_b4": {"verdict": "FAKE", "video_fake_prob": 0.7, "fake_frame_count": 1, "real_frame_count": 0, "total_frames": 1, "top_suspicious_frames": [], "fake_prob_timeline": [0.7]},
    }
    resolved = ensemble._resolve(results)
    expected = (0.9955 * 0.8 + 0.9812 * 0.3 + 0.9 * 0.7) / (0.9955 + 0.9812 + 0.9)
    assert abs(resolved["final_fake_prob"] - expected) < 0.001


def test_comparison_table_sorted_by_auc():
    """Comparison rows should be ordered by AUC descending."""
    ensemble = make_ensemble()
    table = ensemble.get_comparison_table({key: make_results()[key] for key in ("efficientnet_b4", "xception", "vit_base")}, winning_model="xception")
    assert [row["model_key"] for row in table] == ["xception", "vit_base", "efficientnet_b4"]


def test_exactly_one_winner_flag():
    """Exactly one comparison row should be marked as the winner."""
    ensemble = make_ensemble()
    table = ensemble.get_comparison_table({key: make_results()[key] for key in ("efficientnet_b4", "xception", "vit_base")}, winning_model="xception")
    assert sum(1 for row in table if row["is_winner"]) == 1


def test_empty_selection_raises():
    """Empty model selections should be rejected immediately."""
    with pytest.raises(ValueError):
        DeepfakeEnsemble(selected_model_keys=[], device="cpu")
