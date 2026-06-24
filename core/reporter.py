"""Ollama-backed structured authenticity reporting for DeepGuard."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from core.model_registry import get_model_config


LOGGER = logging.getLogger("deepguard.reporter")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
PROJECT_NAME = "DeepGuard"
PROJECT_CONTEXT = (
    "DeepGuard is a multi-model deepfake detection and explanation system built for FCIT, "
    "International Islamic University Islamabad. It samples video frames, crops faces, runs "
    "selected detectors, resolves disagreements with benchmark-aware ensemble logic, generates "
    "GradCAM evidence from the determining model, and prepares a plain-language authenticity report."
)


class LLMReporter:
    """Generate multi-model authenticity reports from DeepGuard analysis results."""

    def __init__(self, ollama_url=OLLAMA_URL, model=OLLAMA_MODEL):
        """
        Initialize the reporter and test the Ollama connection.

        Args:
            ollama_url: REST endpoint for local Ollama generation.
            model: Ollama model name to use.
        """
        self.url = ollama_url
        self.model = model
        self.online = False
        self.connection_error: str | None = None
        try:
            self._verify_ollama_connection()
            self.online = True
        except ConnectionError as exc:
            self.connection_error = str(exc)
            LOGGER.warning("Ollama unavailable during reporter init: %s", exc)

    def _verify_ollama_connection(self) -> None:
        """Ping the local Ollama tags endpoint and raise on failure."""
        tags_url = self.url.replace("/api/generate", "/api/tags")
        try:
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(
                "Unable to reach Ollama at http://localhost:11434. Start it with `ollama serve`."
            ) from exc

    def _build_report_context(self, analysis_result: dict, gradcam_results: list, filename: str) -> dict:
        """Collect the run facts used by both the LLM and the deterministic fallback."""
        winning_model_key = analysis_result["winning_model"]
        winning_model = get_model_config(winning_model_key)
        winning_result = analysis_result["per_model_results"][winning_model_key]
        suspicious_regions = sorted({item["suspicious_region"] for item in gradcam_results if item.get("suspicious_region")})
        resolver_label = (
            "Benchmark-weighted soft voting"
            if analysis_result["resolution_method"] == "auc_weighted_soft_vote"
            else "Majority vote with benchmark-based tiebreaking"
        )

        per_model_rows = []
        for model_key, value in analysis_result["per_model_results"].items():
            config = get_model_config(model_key)
            per_model_rows.append(
                {
                    "model": config["display_name"],
                    "auc": float(config["auc"]),
                    "benchmark_score": config["score_display"],
                    "speed": config["speed"],
                    "verdict": value["verdict"],
                    "video_fake_probability_pct": round(float(value["video_fake_prob"]) * 100, 1),
                    "flagged_frames": f"{value['fake_frame_count']}/{value['total_frames']}",
                    "is_determining_model": model_key == winning_model_key,
                }
            )
        per_model_rows.sort(key=lambda row: row["auc"], reverse=True)

        evidence_highlights = []
        for item in gradcam_results[:3]:
            frame_label = item.get("frame_idx", "N/A")
            region = item.get("suspicious_region", "unspecified facial region")
            fake_prob = item.get("fake_prob")
            if fake_prob is None:
                evidence_highlights.append(f"Frame {frame_label} highlighted the {region}.")
            else:
                evidence_highlights.append(
                    f"Frame {frame_label} highlighted the {region} with {float(fake_prob) * 100:.1f}% fake probability "
                    f"from {winning_model['display_name']}."
                )

        if not evidence_highlights:
            evidence_highlights.append("No GradCAM evidence was available for this run.")

        return {
            "filename": filename,
            "verdict": analysis_result["final_verdict"],
            "final_fake_probability_pct": round(float(analysis_result["final_fake_prob"]) * 100, 1),
            "winning_model_name": winning_model["display_name"],
            "winning_model_score_display": winning_model["score_display"],
            "winning_model_score": float(analysis_result["winning_model_score"]),
            "winning_fake_frames": int(winning_result["fake_frame_count"]),
            "winning_total_frames": int(winning_result["total_frames"]),
            "num_models_used": int(analysis_result["num_models_used"]),
            "agreement": bool(analysis_result["agreement"]),
            "fake_vote_count": int(analysis_result["fake_vote_count"]),
            "real_vote_count": int(analysis_result["real_vote_count"]),
            "resolution_method": analysis_result["resolution_method"],
            "resolution_explanation": analysis_result["resolution_explanation"],
            "resolver_label": resolver_label,
            "frame_count_extracted": int(analysis_result.get("frame_count_extracted", winning_result["total_frames"])),
            "face_count_detected": int(analysis_result.get("face_count_detected", winning_result["total_frames"])),
            "suspicious_regions": suspicious_regions,
            "regions_summary": ", ".join(suspicious_regions) if suspicious_regions else "not determined",
            "per_model_rows": per_model_rows,
            "evidence_highlights": evidence_highlights,
        }

    def _reliability_label(self, benchmark_score: float) -> str:
        """Map the published benchmark score to a human-readable reliability band."""
        if benchmark_score >= 0.98:
            return "very strong"
        if benchmark_score >= 0.94:
            return "strong"
        if benchmark_score >= 0.88:
            return "moderate"
        return "limited"

    def _fallback_sections(self, context: dict) -> dict:
        """Build a polished deterministic report when Ollama is offline or malformed."""
        verdict = context["verdict"]
        fake_pct = context["final_fake_probability_pct"]
        reliability = self._reliability_label(context["winning_model_score"])
        consensus = (
            f"All {context['num_models_used']} selected models agreed with the final verdict."
            if context["agreement"]
            else (
                f"{max(context['fake_vote_count'], context['real_vote_count'])} of {context['num_models_used']} models "
                f"supported the final {verdict} decision."
            )
        )
        project_overview = (
            f"{PROJECT_NAME} is a multi-model deepfake detection project that samples video frames, crops faces, "
            f"runs each selected detector, resolves the decision with {context['resolver_label'].lower()}, and "
            f"adds GradCAM evidence from the determining model. For this run it reviewed {context['frame_count_extracted']} "
            f"sampled frames, detected {context['face_count_detected']} face crops, and used {context['winning_model_name']} "
            f"as the deciding reference."
        )

        if verdict == "FAKE":
            recommendation = [
                "Treat the clip as suspicious until an independent forensic review confirms or rejects manipulation.",
                "Preserve the original file, source link, and chain-of-custody details before sharing the clip as evidence.",
                "If the decision matters for grading, research, or discipline, compare this result with metadata, audio, and source verification.",
            ]
        else:
            recommendation = [
                "No strong manipulation signal was found in this run, but high-stakes use should still include normal source verification.",
                "Keep the original file and metadata so the result can be reproduced if questions arise later.",
                "If the case still feels suspicious, re-run DeepGuard with more models or lower frame skipping for a stricter review.",
            ]

        return {
            "headline": f"{PROJECT_NAME} authenticity result: {verdict}",
            "executive_summary": (
                f"{PROJECT_NAME} assessed {context['filename']} as {verdict} with a final fake probability of "
                f"{fake_pct:.1f}%. {context['winning_model_name']} served as the determining model after DeepGuard "
                f"compared {context['num_models_used']} detector outputs."
            ),
            "project_overview": project_overview,
            "technical_findings": [
                f"{context['winning_fake_frames']} of {context['winning_total_frames']} face frames were flagged as fake by {context['winning_model_name']}.",
                f"Most suspicious visual regions: {context['regions_summary']}.",
                f"DeepGuard processed {context['frame_count_extracted']} sampled frames and {context['face_count_detected']} face crops for this case.",
            ],
            "ensemble_analysis": (
                f"{consensus} {context['resolution_explanation']} DeepGuard uses published benchmark strength to decide "
                "which model should represent the final explanation when models disagree."
            ),
            "confidence_assessment": (
                f"The determining model carries a {reliability} published benchmark profile "
                f"({context['winning_model_score_display']}). That makes this result more dependable than a single unranked "
                "model output, though it should still be paired with external verification for sensitive decisions."
            ),
            "recommended_actions": recommendation,
            "plain_language_brief": (
                f"In simple terms, {PROJECT_NAME} found this video {('likely manipulated' if verdict == 'FAKE' else 'not clearly manipulated')} "
                f"after comparing several detectors and checking the most suspicious face regions."
            ),
        }

    def _build_prompt(self, context: dict) -> str:
        """Create a project-aware prompt that asks Ollama for a strict JSON report."""
        llm_context = {
            "project_name": PROJECT_NAME,
            "project_context": PROJECT_CONTEXT,
            "analysis_case": {
                "file": context["filename"],
                "final_verdict": context["verdict"],
                "final_fake_probability_pct": context["final_fake_probability_pct"],
                "determining_model": context["winning_model_name"],
                "determining_model_benchmark": context["winning_model_score_display"],
                "models_used": context["num_models_used"],
                "agreement": context["agreement"],
                "fake_votes": context["fake_vote_count"],
                "real_votes": context["real_vote_count"],
                "resolver": context["resolver_label"],
                "resolution_explanation": context["resolution_explanation"],
                "sampled_frames": context["frame_count_extracted"],
                "face_crops_detected": context["face_count_detected"],
                "winning_model_flagged_frames": f"{context['winning_fake_frames']}/{context['winning_total_frames']}",
                "suspicious_regions": context["suspicious_regions"] or ["not determined"],
            },
            "per_model_results": context["per_model_rows"],
            "visual_evidence": context["evidence_highlights"],
        }

        return (
            "You are the report-writing component of DeepGuard, a university deepfake detection project. "
            "Write for a non-technical reviewer, but stay precise and grounded in the supplied facts.\n\n"
            "Rules:\n"
            "- Use only the provided information.\n"
            "- Do not invent datasets, metrics, model behavior, or suspicious regions.\n"
            "- Mention DeepGuard by name and explain the project workflow in plain language.\n"
            "- Keep the tone professional, confident, and easy to present in a project demo.\n"
            "- Return only valid JSON with no markdown fences, no leading text, and no trailing text.\n"
            "- `technical_findings` must contain exactly 3 short bullet strings.\n"
            "- `recommended_actions` must contain exactly 3 short bullet strings.\n\n"
            "Return this exact JSON shape:\n"
            "{\n"
            '  "headline": "string",\n'
            '  "executive_summary": "string",\n'
            '  "project_overview": "string",\n'
            '  "technical_findings": ["string", "string", "string"],\n'
            '  "ensemble_analysis": "string",\n'
            '  "confidence_assessment": "string",\n'
            '  "recommended_actions": ["string", "string", "string"],\n'
            '  "plain_language_brief": "string"\n'
            "}\n\n"
            f"DeepGuard case data:\n{json.dumps(llm_context, indent=2)}"
        )

    def _extract_json_blob(self, text: str) -> str | None:
        """Extract the first JSON object from an LLM response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else None

    def _normalize_list(self, value: Any, fallback: list[str], desired_length: int) -> list[str]:
        """Convert a possibly malformed LLM field into a fixed-size list of strings."""
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            cleaned = [segment.strip(" -•") for segment in value.splitlines() if segment.strip()]
        else:
            cleaned = []

        if len(cleaned) < desired_length:
            cleaned.extend(fallback[len(cleaned) : desired_length])
        return cleaned[:desired_length]

    def _normalize_payload(self, payload: dict[str, Any], context: dict, source: str) -> dict:
        """Guarantee that all required sections exist before the UI consumes them."""
        fallback = self._fallback_sections(context)
        normalized = {}
        scalar_fields = [
            "headline",
            "executive_summary",
            "project_overview",
            "ensemble_analysis",
            "confidence_assessment",
            "plain_language_brief",
        ]
        list_fields = {
            "technical_findings": 3,
            "recommended_actions": 3,
        }

        for key in scalar_fields:
            value = payload.get(key) if isinstance(payload, dict) else None
            normalized[key] = str(value).strip() if value is not None and str(value).strip() else fallback[key]

        for key, desired_length in list_fields.items():
            normalized[key] = self._normalize_list(payload.get(key), fallback[key], desired_length)

        normalized["source"] = source
        normalized["report_engine"] = self.model if source == "llm" else "DeepGuard template"
        normalized["evidence_highlights"] = list(context["evidence_highlights"])
        normalized["plain_text"] = self._render_plain_text_report(normalized, context)
        normalized["markdown"] = self._render_markdown_report(normalized, context)
        return normalized

    def _render_plain_text_report(self, payload: dict, context: dict) -> str:
        """Create a copy-friendly plain-text report from the structured payload."""
        sections = [
            f"{PROJECT_NAME} AUTHENTICITY REPORT",
            "=" * 32,
            f"File: {context['filename']}",
            f"Verdict: {context['verdict']} ({context['final_fake_probability_pct']:.1f}% fake probability)",
            "",
            "1. EXECUTIVE SUMMARY",
            payload["executive_summary"],
            "",
            "2. PROJECT OVERVIEW",
            payload["project_overview"],
            "",
            "3. TECHNICAL FINDINGS",
            *[f"- {item}" for item in payload["technical_findings"]],
            "",
            "4. ENSEMBLE ANALYSIS",
            payload["ensemble_analysis"],
            "",
            "5. CONFIDENCE ASSESSMENT",
            payload["confidence_assessment"],
            "",
            "6. RECOMMENDED ACTIONS",
            *[f"- {item}" for item in payload["recommended_actions"]],
            "",
            "7. PLAIN-LANGUAGE BRIEF",
            payload["plain_language_brief"],
        ]
        return "\n".join(sections).strip()

    def _render_markdown_report(self, payload: dict, context: dict) -> str:
        """Create a markdown export of the structured report."""
        technical_findings = "\n".join(f"- {item}" for item in payload["technical_findings"])
        actions = "\n".join(f"- {item}" for item in payload["recommended_actions"])
        return (
            f"# {payload['headline']}\n\n"
            f"**File:** {context['filename']}  \n"
            f"**Verdict:** {context['verdict']} ({context['final_fake_probability_pct']:.1f}% fake probability)\n\n"
            "## Executive Summary\n"
            f"{payload['executive_summary']}\n\n"
            "## Project Overview\n"
            f"{payload['project_overview']}\n\n"
            "## Technical Findings\n"
            f"{technical_findings}\n\n"
            "## Ensemble Analysis\n"
            f"{payload['ensemble_analysis']}\n\n"
            "## Confidence Assessment\n"
            f"{payload['confidence_assessment']}\n\n"
            "## Recommended Actions\n"
            f"{actions}\n\n"
            "## Plain-Language Brief\n"
            f"{payload['plain_language_brief']}\n"
        ).strip()

    def _fallback_report(self, context: dict) -> dict:
        """Return a structured template report when Ollama is unavailable or malformed."""
        return self._normalize_payload(self._fallback_sections(context), context, source="template")

    def generate_report(self, analysis_result: dict, gradcam_results: list, video_filename: str) -> dict:
        """
        Build a project-aware structured report, preferring Ollama and falling back gracefully.

        Args:
            analysis_result: DeepGuard ensemble output bundle.
            gradcam_results: Winning-model GradCAM summaries.
            video_filename: Uploaded source filename.

        Returns:
            Structured report payload for the UI and export actions.
        """
        context = self._build_report_context(analysis_result, gradcam_results, video_filename)
        prompt = self._build_prompt(context)

        if not self.online:
            return self._fallback_report(context)

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                    },
                },
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            raw_response = data.get("response", "").strip()
            json_blob = self._extract_json_blob(raw_response)
            if not json_blob:
                raise ValueError("Ollama response did not contain a JSON object.")
            payload = json.loads(json_blob)
            return self._normalize_payload(payload, context, source="llm")
        except (ValueError, requests.RequestException, json.JSONDecodeError) as exc:
            LOGGER.warning("Falling back to template report after Ollama error: %s", exc)
            self.online = False
            self.connection_error = str(exc)
            return self._fallback_report(context)
