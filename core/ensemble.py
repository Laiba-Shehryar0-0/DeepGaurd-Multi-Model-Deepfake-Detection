"""Multi-model orchestration and benchmark-based conflict resolution for DeepGuard."""

from __future__ import annotations

import logging

from core.aggregator import FrameAggregator
from core.detector import DeepfakeDetector
from core.model_registry import MODEL_REGISTRY, get_available_models, get_model_config


LOGGER = logging.getLogger("deepguard.ensemble")


class DeepfakeEnsemble:
    """Manage selected detectors and resolve their final video-level verdict."""

    def __init__(self, selected_model_keys: list[str], device: str, use_weighted_voting: bool = False):
        """
        Validate the requested detector keys, ensure their weight files exist, and load them.

        Args:
            selected_model_keys: Registry keys chosen by the user.
            device: Preferred inference device for all loaded models.
            use_weighted_voting: Whether to enable benchmark-weighted soft voting.
        """
        if not selected_model_keys:
            raise ValueError("At least one model must be selected.")

        unknown = [key for key in selected_model_keys if key not in MODEL_REGISTRY]
        if unknown:
            raise KeyError(f"Unknown model key(s): {', '.join(unknown)}")

        available_models = get_available_models()
        missing = [key for key in selected_model_keys if key not in available_models]
        if missing:
            readable_names = ", ".join(get_model_config(key)["display_name"] for key in missing)
            raise FileNotFoundError(
                f"Missing weight files for: {readable_names}. Run `python models/download_all.py` first."
            )

        self.selected_model_keys = list(selected_model_keys)
        self.device = device
        self.use_weighted_voting = use_weighted_voting
        self.detectors: dict[str, DeepfakeDetector] = {}
        self.runtime_warnings: list[str] = []

        for model_key in self.selected_model_keys:
            detector = DeepfakeDetector(model_key=model_key, device=device)
            self.detectors[model_key] = detector

    def run(self, face_data: list[dict], preprocessor, progress_callback=None) -> dict:
        """
        Run every selected detector, aggregate its frame-level outputs, and resolve the ensemble verdict.

        Args:
            face_data: Face metadata records from `VideoPreprocessor.detect_and_crop_faces`.
            preprocessor: Shared `VideoPreprocessor` instance for input-size-aware tensor conversion.
            progress_callback: Optional callable receiving `(model_key, completed, total)`.

        Returns:
            Complete ensemble result bundle with per-model outputs, vote counts, and winner metadata.
        """
        per_model_results: dict[str, dict] = {}
        per_model_frame_results: dict[str, list[dict]] = {}
        aggregator = FrameAggregator()
        total_models = len(self.detectors)

        for index, (model_key, detector) in enumerate(self.detectors.items(), start=1):
            LOGGER.info("Running detector `%s` across %s face frames.", model_key, len(face_data))
            if progress_callback:
                progress_callback(model_key, index - 1, total_models)

            tensors = []
            for item in face_data:
                tensors_by_size = item.setdefault("tensors_by_size", {})
                cache_key = detector.preprocess_cache_key
                if cache_key not in tensors_by_size:
                    tensors_by_size[cache_key] = preprocessor.preprocess_face(
                        item["face"],
                        input_size=detector.input_size,
                        mean=detector.normalize_mean,
                        std=detector.normalize_std,
                    )
                tensors.append(tensors_by_size[cache_key])

            frame_results = detector.predict_batch(tensors)
            for item, result in zip(face_data, frame_results):
                result["frame_idx"] = item["frame_idx"]
                result["bbox"] = item["bbox"]

            aggregated = aggregator.aggregate(frame_results)
            aggregated["model_key"] = model_key
            aggregated["display_name"] = detector.config["display_name"]
            aggregated["benchmark"] = detector.config["benchmark_label"]
            aggregated["auc"] = detector.auc
            aggregated["score_display"] = detector.config["score_display"]
            aggregated["speed"] = detector.config["speed"]
            aggregated["is_generic_fallback"] = detector.is_generic_fallback
            aggregated["needs_fine_tuning"] = detector.needs_fine_tuning
            aggregated["weight_source"] = detector.weight_source
            per_model_results[model_key] = aggregated
            per_model_frame_results[model_key] = frame_results
            self.runtime_warnings.extend(detector.runtime_warnings)

            if progress_callback:
                progress_callback(model_key, index, total_models)

        resolved = self._resolve(per_model_results)
        resolved["per_model_results"] = per_model_results
        resolved["per_model_frame_results"] = per_model_frame_results
        resolved["top_suspicious_frames"] = per_model_results[resolved["winning_model"]]["top_suspicious_frames"]
        resolved["comparison_table"] = self.get_comparison_table(per_model_results, resolved["winning_model"])
        return resolved

    def _select_highest_auc(self, model_keys: list[str], per_model_results: dict) -> str:
        """Select the highest-AUC model, using video_fake_prob as the secondary tiebreaker."""
        return max(
            model_keys,
            key=lambda key: (
                MODEL_REGISTRY[key]["auc"],
                per_model_results[key]["video_fake_prob"],
            ),
        )

    def _weighted_soft_vote(self, per_model_results: dict) -> dict:
        """
        Compute the benchmark-weighted average fake probability across all selected models.

        Args:
            per_model_results: Model-keyed aggregated results.

        Returns:
            Partial resolution dict for weighted soft voting.
        """
        total_weight = sum(MODEL_REGISTRY[key]["auc"] for key in per_model_results)
        weighted_fake_prob = sum(
            MODEL_REGISTRY[key]["auc"] * per_model_results[key]["video_fake_prob"]
            for key in per_model_results
        ) / total_weight
        final_verdict = "FAKE" if weighted_fake_prob > 0.5 else "REAL"
        agreeing = [key for key, value in per_model_results.items() if value["verdict"] == final_verdict]
        if not agreeing:
            agreeing = list(per_model_results)
        winning_model = self._select_highest_auc(agreeing, per_model_results)
        return {
            "final_verdict": final_verdict,
            "final_fake_prob": weighted_fake_prob,
            "winning_model": winning_model,
            "resolution_method": "auc_weighted_soft_vote",
            "agreement": all(value["verdict"] == final_verdict for value in per_model_results.values()),
        }

    def _resolve(self, per_model_results: dict) -> dict:
        """
        Resolve model disagreement using the benchmark-based ensemble rules.

        Args:
            per_model_results: Mapping of model_key to aggregated video-level result.

        Returns:
            Fully assembled ensemble output dictionary.
        """
        n = len(per_model_results)

        if n == 1:
            key = list(per_model_results.keys())[0]
            result = per_model_results[key]
            return self._build_output(
                per_model_results=per_model_results,
                final_verdict=result["verdict"],
                final_fake_prob=result["video_fake_prob"],
                winning_model=key,
                resolution_method="single",
                agreement=True,
            )

        if self.use_weighted_voting:
            weighted = self._weighted_soft_vote(per_model_results)
            return self._build_output(per_model_results=per_model_results, **weighted)

        fake_voters = [key for key, value in per_model_results.items() if value["verdict"] == "FAKE"]
        real_voters = [key for key, value in per_model_results.items() if value["verdict"] == "REAL"]

        if len(fake_voters) == n or len(real_voters) == n:
            winning_model = self._select_highest_auc(list(per_model_results.keys()), per_model_results)
            return self._build_output(
                per_model_results=per_model_results,
                final_verdict=per_model_results[winning_model]["verdict"],
                final_fake_prob=per_model_results[winning_model]["video_fake_prob"],
                winning_model=winning_model,
                resolution_method="unanimous",
                agreement=True,
            )

        if len(fake_voters) != len(real_voters):
            majority_side = "FAKE" if len(fake_voters) > len(real_voters) else "REAL"
            majority_voters = fake_voters if majority_side == "FAKE" else real_voters
            winning_model = self._select_highest_auc(majority_voters, per_model_results)
            return self._build_output(
                per_model_results=per_model_results,
                final_verdict=majority_side,
                final_fake_prob=per_model_results[winning_model]["video_fake_prob"],
                winning_model=winning_model,
                resolution_method="majority_vote",
                agreement=False,
            )

        winning_model = self._select_highest_auc(list(per_model_results.keys()), per_model_results)
        return self._build_output(
            per_model_results=per_model_results,
            final_verdict=per_model_results[winning_model]["verdict"],
            final_fake_prob=per_model_results[winning_model]["video_fake_prob"],
            winning_model=winning_model,
            resolution_method="tie_broken_by_auc",
            agreement=False,
        )

    def _build_output(
        self,
        per_model_results: dict,
        final_verdict: str,
        final_fake_prob: float,
        winning_model: str,
        resolution_method: str,
        agreement: bool,
    ) -> dict:
        """
        Assemble the final ensemble output dictionary.

        Args:
            per_model_results: Model-keyed aggregated results.
            final_verdict: Final REAL/FAKE verdict chosen by the ensemble.
            final_fake_prob: Final fake probability associated with that verdict.
            winning_model: Model key chosen as the determining model.
            resolution_method: Resolver branch name.
            agreement: Whether all models agreed with the final verdict.

        Returns:
            Final DeepGuard ensemble result bundle for the UI and reporter.
        """
        fake_vote_count = sum(1 for value in per_model_results.values() if value["verdict"] == "FAKE")
        real_vote_count = sum(1 for value in per_model_results.values() if value["verdict"] == "REAL")
        output = {
            "final_verdict": final_verdict,
            "final_fake_prob": float(final_fake_prob),
            "final_confidence_pct": round(float(final_fake_prob) * 100, 1),
            "winning_model": winning_model,
            "winning_model_auc": float(MODEL_REGISTRY[winning_model]["auc"]),
            "winning_model_score": float(MODEL_REGISTRY[winning_model]["auc"]),
            "winning_model_score_display": get_model_config(winning_model)["score_display"],
            "resolution_method": resolution_method,
            "agreement": agreement,
            "num_models_used": len(per_model_results),
            "fake_vote_count": fake_vote_count,
            "real_vote_count": real_vote_count,
            "fake_prob_timelines": {
                key: value["fake_prob_timeline"] for key, value in per_model_results.items()
            },
        }
        output["resolution_explanation"] = self._resolution_explanation(
            resolution_method=resolution_method,
            winning_model=winning_model,
            per_model_results=per_model_results,
            final_verdict=final_verdict,
            final_fake_prob=final_fake_prob,
            fake_vote_count=fake_vote_count,
            real_vote_count=real_vote_count,
        )
        return output

    def _resolution_explanation(
        self,
        resolution_method: str,
        winning_model: str,
        per_model_results: dict,
        final_verdict: str,
        final_fake_prob: float,
        fake_vote_count: int,
        real_vote_count: int,
    ) -> str:
        """
        Explain in plain English how the ensemble selected the determining model.

        Args:
            resolution_method: Resolver branch name.
            winning_model: Model key that determined the final verdict.
            per_model_results: Model-keyed aggregated results.
            final_verdict: Final REAL/FAKE verdict.
            final_fake_prob: Final fake probability.
            fake_vote_count: Number of FAKE voters.
            real_vote_count: Number of REAL voters.

        Returns:
            Human-readable explanation string for the UI and report.
        """
        config = get_model_config(winning_model)
        name = config["display_name"]
        score_display = config["score_display"]
        if resolution_method == "single":
            return f"Single model analysis by {name}."
        if resolution_method == "unanimous":
            return (
                f"All {len(per_model_results)} models agreed. {name} ({score_display}) "
                "used as the primary reference due to the highest published benchmark score."
            )
        if resolution_method == "majority_vote":
            majority = max(fake_vote_count, real_vote_count)
            return (
                f"{majority} of {len(per_model_results)} models voted {final_verdict}. "
                f"{name} selected among majority voters by highest published benchmark score ({score_display})."
            )
        if resolution_method == "tie_broken_by_auc":
            return (
                f"Models split equally ({fake_vote_count}v{real_vote_count}). {name} "
                f"({score_display}) selected as the tiebreaker by published benchmark rank."
            )
        if resolution_method == "auc_weighted_soft_vote":
            return (
                f"Benchmark-weighted average fake probability: {final_fake_prob:.1%}. "
                f"Winner by published benchmark rank: {name}."
            )
        return f"{name} selected by {resolution_method}."

    def get_comparison_table(self, per_model_results: dict, winning_model: str) -> list[dict]:
        """
        Build the UI comparison table sorted by benchmark score descending.

        Args:
            per_model_results: Model-keyed aggregated results.
            winning_model: Model key chosen by the resolver.

        Returns:
            List of comparison-row dictionaries for Streamlit.
        """
        rows = []
        for model_key, result in per_model_results.items():
            config = get_model_config(model_key)
            rows.append(
                {
                    "model_key": model_key,
                    "display_name": config["display_name"],
                    "verdict": result["verdict"],
                    "fake_prob_pct": float(result["video_fake_prob"] * 100),
                    "fake_frames": int(result["fake_frame_count"]),
                    "total_frames": int(result["total_frames"]),
                    "auc": float(config["auc"]),
                    "score_display": config["score_display"],
                    "benchmark": config["benchmark_label"],
                    "speed": config["speed"],
                    "is_winner": model_key == winning_model,
                }
            )
        return sorted(rows, key=lambda row: row["auc"], reverse=True)
