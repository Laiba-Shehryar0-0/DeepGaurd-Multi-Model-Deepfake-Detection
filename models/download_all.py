"""Download all DeepGuard model weights and print a summary table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.model_registry import MODEL_REGISTRY
from models.download_weights import configure_logging, download_model


def _status_icon(status: str) -> str:
    """Return a readable status glyph for the download summary."""
    return {
        "READY": "✅ Ready",
        "FALLBACK": "⚠️ Fallback",
        "FAILED": "❌ Failed",
    }.get(status, status)


def main() -> None:
    """Download every registered model sequentially and print a summary."""
    configure_logging()
    results = [download_model(model_key) for model_key in sorted(MODEL_REGISTRY, key=lambda key: MODEL_REGISTRY[key]["auc"], reverse=True)]

    print("Model                    Status        File                                  Score")
    print("──────────────────────────────────────────────────────────────────────────────────────────────")
    for result in results:
        print(
            f"{result['display_name']:<24} {_status_icon(result['status']):<13} "
            f"{result['weight_file']:<36} {MODEL_REGISTRY[result['model_key']]['score_display']}"
        )
        if result["notes"]:
            print(f"  note: {result['notes']}")


if __name__ == "__main__":
    main()
