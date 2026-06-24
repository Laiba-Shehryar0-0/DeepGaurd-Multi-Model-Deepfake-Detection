"""One-command bootstrapper for installing and launching DeepGuard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib import error, request


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
INSTALL_STAMP = VENV_DIR / ".deepguard_requirements_installed"
DEFAULT_PORT = 8501
OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "llama3.2"


def _step(message: str) -> None:
    """Print a readable progress message."""
    print(f"\n==> {message}", flush=True)


def _venv_python() -> Path:
    """Return the Python executable path inside the project virtual environment."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    cwd: Path = ROOT_DIR,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and stream its output to the current terminal."""
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        check=check,
        text=True,
    )


def _require_supported_python() -> None:
    """Fail fast if the current interpreter is too old."""
    if sys.version_info < (3, 10):
        raise SystemExit("DeepGuard requires Python 3.10 or newer.")


def _create_venv_if_needed() -> None:
    """Create the local virtual environment when it does not exist yet."""
    python_path = _venv_python()
    if python_path.exists():
        return
    _step("Creating project virtual environment")
    _run([sys.executable, "-m", "venv", str(VENV_DIR)])


def _install_requirements(force: bool = False) -> None:
    """Install all Python dependencies into the project virtual environment."""
    python_path = _venv_python()
    if not force and INSTALL_STAMP.exists() and INSTALL_STAMP.stat().st_mtime >= REQUIREMENTS_FILE.stat().st_mtime:
        _step("Using existing Python environment")
        return
    _step("Installing Python dependencies")
    _run([str(python_path), "-m", "ensurepip", "--upgrade"])
    _run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python_path), "-m", "pip", "install", "-r", "requirements.txt"])
    INSTALL_STAMP.touch()


def _run_environment_check() -> None:
    """Run DeepGuard's built-in environment verification script."""
    python_path = _venv_python()
    _step("Running environment checks")
    _run([str(python_path), "check_env.py"])


def _download_models() -> None:
    """Download or reuse the registered DeepGuard model files."""
    python_path = _venv_python()
    _step("Preparing model weights")
    _run([str(python_path), "models/download_all.py"])


def _ollama_tags() -> list[str]:
    """Return the list of local Ollama model tags, or an empty list when offline."""
    try:
        with request.urlopen(OLLAMA_URL, timeout=5) as response:
            if response.status != 200:
                return []
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    models = payload.get("models", [])
    tags: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str):
                tags.append(name)
    return tags


def _ollama_reachable() -> bool:
    """Check whether the local Ollama API is currently reachable."""
    return bool(_ollama_tags())


def _start_ollama_server() -> bool:
    """Start `ollama serve` in the background when the binary is available."""
    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        return False

    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([ollama_bin, "serve"], **kwargs)
    return True


def _wait_for_ollama(timeout_seconds: int = 20) -> bool:
    """Poll the local Ollama endpoint until it comes online or times out."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _ollama_tags():
            return True
        time.sleep(1)
    return False


def _ensure_ollama_model() -> None:
    """Ensure Ollama is reachable and the expected local model tag exists."""
    tags = _ollama_tags()
    if not tags:
        _step("Starting Ollama server")
        if _start_ollama_server() and _wait_for_ollama():
            tags = _ollama_tags()

    if not tags:
        print(
            "Warning: Ollama is not available. DeepGuard will still run, "
            "but report generation will use the fallback template.",
            flush=True,
        )
        print("Install Ollama from https://ollama.com/download and run `ollama pull llama3.2` for full report support.", flush=True)
        return

    if not any(tag == OLLAMA_MODEL or tag.startswith(f"{OLLAMA_MODEL}:") for tag in tags):
        ollama_bin = shutil.which("ollama")
        if ollama_bin is None:
            print(f"Warning: Ollama is online but `{OLLAMA_MODEL}` is missing and the `ollama` CLI was not found.", flush=True)
            return
        _step(f"Pulling Ollama model `{OLLAMA_MODEL}`")
        _run([ollama_bin, "pull", OLLAMA_MODEL])


def _launch_streamlit(port: int, open_browser: bool) -> None:
    """Start the Streamlit app in the project virtual environment."""
    python_path = _venv_python()
    env = dict(os.environ)
    if not open_browser:
        env["BROWSER"] = "none"

    _step(f"Launching DeepGuard on http://localhost:{port}")
    _run(
        [
            str(python_path),
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "false",
        ],
        env=env,
    )


def parse_args() -> argparse.Namespace:
    """Parse bootstrap command-line options."""
    parser = argparse.ArgumentParser(description="Install and launch DeepGuard with one command.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the Streamlit app.")
    parser.add_argument("--skip-model-downloads", action="store_true", help="Do not run models/download_all.py.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip check_env.py after installing dependencies.")
    parser.add_argument("--skip-ollama", action="store_true", help="Do not attempt to start or pull Ollama.")
    parser.add_argument("--no-browser", action="store_true", help="Do not request automatic browser opening.")
    parser.add_argument("--reinstall", action="store_true", help="Force reinstall Python requirements even if .venv is already prepared.")
    return parser.parse_args()


def main() -> None:
    """Create the runtime environment and launch DeepGuard."""
    args = parse_args()
    _require_supported_python()
    _create_venv_if_needed()
    _install_requirements(force=args.reinstall)

    if not args.skip_checks:
        _run_environment_check()
    if not args.skip_model_downloads:
        _download_models()
    if not args.skip_ollama:
        _ensure_ollama_model()

    _launch_streamlit(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDeepGuard launch cancelled by user.", flush=True)
        raise SystemExit(130)
