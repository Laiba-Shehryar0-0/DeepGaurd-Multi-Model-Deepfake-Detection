"""Environment verification script for DeepGuard v2.0."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Iterable
from urllib import error, request


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

PACKAGE_IMPORTS: Iterable[tuple[str, str]] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("transformers", "transformers"),
    ("facenet-pytorch", "facenet_pytorch"),
    ("grad-cam", "pytorch_grad_cam"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("albumentations", "albumentations"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("streamlit", "streamlit"),
    ("requests", "requests"),
    ("Pillow", "PIL"),
    ("tqdm", "tqdm"),
    ("python-dotenv", "dotenv"),
    ("huggingface_hub", "huggingface_hub"),
    ("einops", "einops"),
    ("pytest", "pytest"),
)


def _status_prefix(ok: bool) -> str:
    """Return a colored pass/fail glyph."""
    return CHECK if ok else CROSS


def verify_imports() -> bool:
    """Import each required package and print a status line."""
    all_ok = True
    for package_name, module_name in PACKAGE_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"{_status_prefix(True)} {package_name}", flush=True)
        except Exception as exc:  # pragma: no cover - environment-specific failures
            all_ok = False
            print(f"{_status_prefix(False)} {package_name}: {exc}", flush=True)
    return all_ok


def verify_ollama() -> bool:
    """Ping the local Ollama tags endpoint."""
    try:
        with request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            if response.status != 200:
                return False
            json.loads(response.read().decode("utf-8"))
        return True
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return False


def list_model_files() -> list[Path]:
    """Return the model files currently present on disk."""
    patterns = ("*.pth", "*.pt", "*.safetensors", "*.bin", "*.h5")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(MODELS_DIR.glob(pattern)))
    return sorted(set(files))


def print_model_status() -> None:
    """Print the currently available model weight files."""
    print("Model files present:", flush=True)
    files = list_model_files()
    if not files:
        print("  - none", flush=True)
        return
    for path in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.relative_to(ROOT_DIR)} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    imports_ok = verify_imports()
    ollama_ok = verify_ollama()
    print(f'Ollama reachable: {"YES" if ollama_ok else "NO"}', flush=True)
    print_model_status()
    raise SystemExit(0 if imports_ok else 1)
