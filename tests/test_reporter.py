"""Tests for Ollama-backed report generation and fallback behavior."""

from __future__ import annotations

import requests

from core.reporter import LLMReporter


ENSEMBLE_RESULT = {
    "final_verdict": "FAKE",
    "final_fake_prob": 0.91,
    "winning_model": "xception",
    "winning_model_auc": 0.9995,
    "winning_model_score": 0.9995,
    "winning_model_score_display": "99.33% acc / AUC 0.9995",
    "resolution_method": "majority_vote",
    "resolution_explanation": "2 of 3 models voted FAKE. Xception (FaceForge) selected among majority voters by highest published benchmark score (99.33% acc / AUC 0.9995).",
    "agreement": False,
    "num_models_used": 3,
    "fake_vote_count": 2,
    "real_vote_count": 1,
    "per_model_results": {
        "xception": {
            "verdict": "FAKE",
            "video_fake_prob": 0.91,
            "fake_frame_count": 12,
            "total_frames": 15,
        },
        "vit_base": {
            "verdict": "FAKE",
            "video_fake_prob": 0.88,
            "fake_frame_count": 11,
            "total_frames": 15,
        },
        "efficientnet_b4": {
            "verdict": "REAL",
            "video_fake_prob": 0.31,
            "fake_frame_count": 6,
            "total_frames": 15,
        },
    },
}
GRADCAM_RESULTS = [
    {"suspicious_region": "left jaw"},
    {"suspicious_region": "nose bridge"},
]


def test_ollama_unreachable_fallback(monkeypatch):
    """Reporter should return a structured fallback report when Ollama is offline."""

    def raise_connection(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("core.reporter.requests.get", raise_connection)
    reporter = LLMReporter()
    report = reporter.generate_report(ENSEMBLE_RESULT, GRADCAM_RESULTS, "sample.mp4")
    assert report["source"] == "template"
    assert report["headline"]
    assert report["plain_text"]
    assert len(report["technical_findings"]) == 3


def test_report_contains_verdict(monkeypatch):
    """Generated reports should keep the final verdict in the export text."""

    def raise_connection(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("core.reporter.requests.get", raise_connection)
    reporter = LLMReporter()
    report = reporter.generate_report(ENSEMBLE_RESULT, GRADCAM_RESULTS, "sample.mp4")
    assert "FAKE" in report["plain_text"] or "REAL" in report["plain_text"]


def test_valid_llm_json_is_normalized(monkeypatch):
    """Reporter should normalize a valid Ollama JSON response into the UI payload."""

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr("core.reporter.requests.get", lambda *args, **kwargs: DummyResponse({}))
    monkeypatch.setattr(
        "core.reporter.requests.post",
        lambda *args, **kwargs: DummyResponse(
            {
                "response": """
                {
                  "headline": "DeepGuard case review",
                  "executive_summary": "DeepGuard marked the clip as FAKE with high confidence.",
                  "project_overview": "DeepGuard compares several detectors, resolves disagreements, and explains the winner with GradCAM.",
                  "technical_findings": [
                    "12 of 15 frames were flagged as fake.",
                    "Left jaw and nose bridge were the main suspicious regions.",
                    "Three models were included in the final review."
                  ],
                  "ensemble_analysis": "Two of three models voted FAKE, so DeepGuard used the strongest benchmark-backed model from that side.",
                  "confidence_assessment": "The deciding model has a very strong published benchmark profile.",
                  "recommended_actions": [
                    "Treat the clip as suspicious.",
                    "Preserve the original file.",
                    "Use independent verification before relying on it."
                  ],
                  "plain_language_brief": "The video looks manipulated based on both model agreement and highlighted face regions."
                }
                """
            }
        ),
    )

    reporter = LLMReporter()
    report = reporter.generate_report(ENSEMBLE_RESULT, GRADCAM_RESULTS, "sample.mp4")
    assert report["source"] == "llm"
    assert report["report_engine"] == reporter.model
    assert report["headline"] == "DeepGuard case review"
    assert len(report["recommended_actions"]) == 3
