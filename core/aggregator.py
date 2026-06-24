"""Aggregate frame-level predictions into a video-level verdict."""

from __future__ import annotations


class FrameAggregator:
    """Reduce per-frame predictions to a single video verdict."""

    def aggregate(self, frame_results: list[dict]) -> dict:
        """
        Majority-vote frame predictions and summarize the fake-probability timeline.

        Args:
            frame_results: List of detector outputs for each analyzed face frame.

        Returns:
            Summary dictionary with verdict statistics, top suspicious frames,
            and a fake-probability timeline in frame order.
        """
        if not frame_results:
            raise ValueError("No frame results were provided for aggregation.")

        ordered_results = sorted(frame_results, key=lambda item: item.get("frame_idx", 0))
        fake_frame_count = sum(1 for result in ordered_results if result["label"] == "FAKE")
        real_frame_count = sum(1 for result in ordered_results if result["label"] == "REAL")
        total_frames = len(ordered_results)
        mean_fake_probability = sum(result["fake_prob"] for result in ordered_results) / total_frames
        verdict = "FAKE" if (fake_frame_count / total_frames) > 0.5 else "REAL"

        ranked_results = sorted(
            ordered_results,
            key=lambda item: item["fake_prob"],
            reverse=True,
        )
        top_suspicious_frames = [result.get("frame_idx", index) for index, result in enumerate(ranked_results[:3])]

        return {
            "verdict": verdict,
            "video_fake_prob": float(mean_fake_probability),
            "fake_frame_count": fake_frame_count,
            "real_frame_count": real_frame_count,
            "total_frames": total_frames,
            "top_suspicious_frames": top_suspicious_frames,
            "fake_prob_timeline": [float(result["fake_prob"]) for result in ordered_results],
        }
