# DeepGuard Full Run Guide

This project can now be installed and launched on another laptop with one command from the project root:

```bash
python run_deepguard.py
```

## What This Command Does

`run_deepguard.py` will:

1. create a local `.venv` virtual environment if it does not exist
2. install all Python packages from `requirements.txt`
3. run `check_env.py`
4. download or reuse the registered model files
5. try to start Ollama automatically if it is installed but not already running
6. pull `llama3.2` automatically if Ollama is available but the model is missing
7. launch the Streamlit app

## Minimum Prerequisites

Before running the command on a new laptop, make sure these are available:

- Python `3.10+`
- internet connection for first-time package and model downloads
- Ollama installed if you want the LLM-generated report

Ollama download:

- https://ollama.com/download

If Ollama is not installed, DeepGuard still opens, but it uses the fallback text report instead of the local LLM report.

## First Run

From the project folder:

```bash
python run_deepguard.py
```

Then open:

```text
http://localhost:8501
```

The first run can take a while because it may:

- install PyTorch and other libraries
- download multiple model checkpoints
- pull the `llama3.2` Ollama model

## Useful Options

Run on a different port:

```bash
python run_deepguard.py --port 8502
```

Skip model downloads:

```bash
python run_deepguard.py --skip-model-downloads
```

Skip Ollama setup:

```bash
python run_deepguard.py --skip-ollama
```

Skip environment checks:

```bash
python run_deepguard.py --skip-checks
```

Force a fresh dependency reinstall:

```bash
python run_deepguard.py --reinstall
```

Prevent browser auto-open requests:

```bash
python run_deepguard.py --no-browser
```

## Manual Fallback Path

If you want to run things manually instead of the bootstrap command:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_env.py
.venv/bin/python models/download_all.py
ollama serve
ollama pull llama3.2
.venv/bin/python -m streamlit run app.py
```

On Windows, replace `.venv/bin/python` with:

```text
.venv\Scripts\python.exe
```

## Troubleshooting

If `python run_deepguard.py` fails:

- confirm Python is `3.10+`
- confirm the laptop has enough free disk space for model files
- confirm internet access is available on first run
- if Ollama is missing, install it from `https://ollama.com/download`
- if Streamlit opens but the report says fallback mode, run `ollama pull llama3.2`

## Main Files

- bootstrap runner: [run_deepguard.py](/Users/mm/DeepFake%20Project/run_deepguard.py)
- app entry point: [app.py](/Users/mm/DeepFake%20Project/app.py)
- environment check: [check_env.py](/Users/mm/DeepFake%20Project/check_env.py)
- model downloader: [models/download_all.py](/Users/mm/DeepFake%20Project/models/download_all.py)
- model sources: [docs/model_sources.md](/Users/mm/DeepFake%20Project/docs/model_sources.md)
