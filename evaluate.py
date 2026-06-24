"""Evaluation harness for DeepGuard v2.0 multi-model detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.ensemble import DeepfakeEnsemble
from core.model_registry import get_available_models, get_model_config
from core.preprocessor import VideoPreprocessor


LOGGER = logging.getLogger("deepguard.evaluate")
ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"


def configure_logging() -> None:
    """Configure evaluation logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def collect_videos(dataset_root: Path) -> list[tuple[Path, str]]:
    """Gather videos from the expected `real/` and `fake/` subfolders."""
    collected: list[tuple[Path, str]] = []
    for label in ("real", "fake"):
        label_dir = dataset_root / label
        if not label_dir.exists():
            LOGGER.warning("Missing dataset folder: %s", label_dir)
            continue
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                collected.append((path, label.upper()))
    return collected


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path, title: str) -> None:
    """Render and save one confusion matrix figure."""
    matrix = confusion_matrix(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="mako", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    """Compute the standard binary classification metrics for one model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
    }


def main() -> None:
    """Evaluate all available DeepGuard models individually and as an ensemble."""
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard on a labeled video dataset.")
    parser.add_argument("dataset_root", type=Path, help="Folder containing `real/` and `fake/` subfolders.")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_logging()
    videos = collect_videos(args.dataset_root)
    if not videos:
        raise SystemExit("No videos found under `real/` and `fake/`.")

    available_models = get_available_models()
    if not available_models:
        raise SystemExit("No model weights were found. Run `python models/download_all.py` first.")

    model_keys = list(available_models)
    ensemble = DeepfakeEnsemble(selected_model_keys=model_keys, device=args.device)

    y_true_map: dict[str, list[int]] = {model_key: [] for model_key in model_keys}
    y_pred_map: dict[str, list[int]] = {model_key: [] for model_key in model_keys}
    y_score_map: dict[str, list[float]] = {model_key: [] for model_key in model_keys}
    y_true_map["ensemble"] = []
    y_pred_map["ensemble"] = []
    y_score_map["ensemble"] = []
    per_video_records: list[dict] = []

    for video_path, true_label in videos:
        preprocessor = VideoPreprocessor(device=args.device, frame_skip=args.frame_skip, max_frames=args.max_frames)
        try:
            frames = preprocessor.extract_frames(str(video_path))
            face_data = preprocessor.detect_and_crop_faces(frames)
            result = ensemble.run(face_data=face_data, preprocessor=preprocessor)
        except Exception as exc:
            LOGGER.warning("Skipping %s after pipeline failure: %s", video_path.name, exc)
            continue

        true_value = 1 if true_label == "FAKE" else 0
        for model_key, model_result in result["per_model_results"].items():
            predicted_label = model_result["verdict"]
            confidence = model_result["video_fake_prob"]
            y_true_map[model_key].append(true_value)
            y_pred_map[model_key].append(1 if predicted_label == "FAKE" else 0)
            y_score_map[model_key].append(confidence)
            per_video_records.append(
                {
                    "filename": video_path.name,
                    "mode": model_key,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                }
            )

        final_label = result["final_verdict"]
        final_fake_prob = result["final_fake_prob"]
        y_true_map["ensemble"].append(true_value)
        y_pred_map["ensemble"].append(1 if final_label == "FAKE" else 0)
        y_score_map["ensemble"].append(final_fake_prob)
        per_video_records.append(
            {
                "filename": video_path.name,
                "mode": "ensemble",
                "true_label": true_label,
                "predicted_label": final_label,
                "confidence": final_fake_prob,
            }
        )
        LOGGER.info("%s -> ensemble=%s (%.3f)", video_path.name, final_label, final_fake_prob)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_video_records).to_csv(RESULTS_DIR / "evaluation_report.csv", index=False)

    leaderboard_rows = []
    for model_key in [*model_keys, "ensemble"]:
        if not y_true_map[model_key]:
            continue
        metrics = compute_metrics(y_true_map[model_key], y_pred_map[model_key], y_score_map[model_key])
        leaderboard_rows.append(
            {
                "model": "Ensemble" if model_key == "ensemble" else get_model_config(model_key)["display_name"],
                **metrics,
            }
        )
        title = "DeepGuard Ensemble Confusion Matrix" if model_key == "ensemble" else f"{get_model_config(model_key)['display_name']} Confusion Matrix"
        filename = "confusion_matrix_ensemble.png" if model_key == "ensemble" else f"confusion_matrix_{model_key}.png"
        save_confusion_matrix(y_true_map[model_key], y_pred_map[model_key], RESULTS_DIR / filename, title)

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("f1", ascending=False)
    leaderboard_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("DeepGuard model leaderboard by F1-score:")
    for row in leaderboard_df.itertuples(index=False):
        print(
            f"- {row.model}: F1={row.f1:.4f} | Acc={row.accuracy:.4f} | "
            f"Prec={row.precision:.4f} | Recall={row.recall:.4f} | AUC={row.auc_roc:.4f}"
        )

    ensemble_f1 = float(leaderboard_df.loc[leaderboard_df["model"] == "Ensemble", "f1"].iloc[0]) if "Ensemble" in set(leaderboard_df["model"]) else float("nan")
    individual_f1_values = leaderboard_df.loc[leaderboard_df["model"] != "Ensemble", "f1"]
    if not individual_f1_values.empty:
        outperforms_all = bool((ensemble_f1 > individual_f1_values).all())
        print(f"Ensemble outperforms every individual model: {'YES' if outperforms_all else 'NO'}")


if __name__ == "__main__":
    main()
