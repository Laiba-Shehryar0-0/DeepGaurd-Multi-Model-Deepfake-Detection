# DeepGuard &mdash; Multi-Model Deepfake Detection & Explainability

DeepGuard is a deepfake detection system that runs multiple neural network classifiers on uploaded videos, resolves their verdicts through ensemble logic, and explains the decision with GradCAM heatmaps and a natural-language forensic report. Built as a Streamlit web app with a polished forensic dashboard interface.

---

## How It Works

```
Upload Video  →  Frame Sampling (OpenCV)  →  Face Detection (MTCNN)
                                                    ↓
                                          Model Ensemble (1-6 classifiers)
                                                    ↓
                                    Verdict Resolution (majority vote / weighted)
                                                    ↓
                          GradCAM Explainability  ←  →  LLM Forensic Report (Ollama)
```

1. **Detect** &mdash; OpenCV samples frames from the video, MTCNN crops faces, and each selected model scores every face crop as real or fake
2. **Resolve** &mdash; Ensemble resolver combines model verdicts using majority vote with benchmark-based tiebreaking, or optional benchmark-weighted soft voting
3. **Explain** &mdash; GradCAM generates heatmap overlays on the most suspicious frames, highlighting which facial regions triggered the detection
4. **Report** &mdash; Ollama (llama3.2) writes a structured forensic report with executive summary, technical findings, evidence highlights, and recommended actions

---

## Models

| Model | Benchmark | Speed |
|-------|-----------|-------|
| Xception (FaceForge) | 99.33% acc / AUC 0.9995 (FF++) | Medium |
| ViT-B/16 | 98.70% accuracy | Medium |
| ResNet-50 (FF++) | AUC 0.9450 (FF++) | Fast |
| EfficientNet-B7 | AUC 0.9280 (DFDC) | Slow |
| EfficientNet-B4 | AUC 0.9000 (DFDC) | Fast |
| MesoNet-4 | ~82% (baseline) | Very Fast |

All models run frame-level binary classification on MTCNN-cropped face images. The ensemble resolver picks the final verdict based on published benchmark scores, not per-video confidence.

---

## Features

- **Multi-model ensemble** &mdash; run 1 to 6 classifiers simultaneously and compare their verdicts side by side
- **GradCAM explainability** &mdash; visual heatmaps showing which facial regions the winning model found suspicious
- **LLM forensic reports** &mdash; Ollama generates structured authenticity reports with executive summary, technical findings, and recommended actions
- **Fake probability timeline** &mdash; frame-by-frame confidence chart showing whether suspicion is isolated or persistent
- **Benchmark-weighted voting** &mdash; optional advanced mode that weights each model's vote by its published benchmark score
- **Downloadable reports** &mdash; export results as Markdown or plain text for sharing with non-technical reviewers
- **Model comparison table** &mdash; per-model verdict, fake probability, flagged frame count, and benchmark score in one view
- **One-command setup** &mdash; `python run_deepguard.py` handles venv creation, dependency installation, environment checks, and Streamlit launch

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Interface | Streamlit |
| Deep Learning | PyTorch, timm, torchvision, Transformers |
| Face Detection | MTCNN (facenet-pytorch) |
| Explainability | GradCAM (grad-cam) |
| Video Processing | OpenCV |
| LLM Reports | Ollama (llama3.2) |
| Data | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with `llama3.2` pulled (`ollama pull llama3.2`)
- 4GB+ RAM (8GB recommended)
- GPU optional (CPU mode supported)

### One-Command Launch

```bash
python run_deepguard.py
```

This creates a virtual environment, installs dependencies, checks the environment, downloads model weights, starts Ollama, and launches the Streamlit app.

### Manual Setup

```bash
pip install -r requirements.txt
python models/download_all.py
ollama serve
streamlit run app.py
```

---

## Ensemble Decision Logic

| Scenario | Resolution Rule |
|----------|----------------|
| Single model | That model's verdict is used directly |
| Unanimous | Highest-benchmark model among voters becomes the reference |
| Majority vote | Majority side wins; strongest benchmark model on that side determines the reference |
| Perfect tie | Highest-benchmark model overall breaks the tie |

**Weighted mode** (optional): computes a weighted average fake probability across all selected models, where each weight equals the model's published benchmark score.

---

## Evaluation

```bash
python evaluate.py
```

Runs all available models individually and as an ensemble on a dataset split into `real/` and `fake/` folders. Outputs confusion matrices, per-model metrics, and comparison CSVs to the `results/` directory.

---

## Project Structure

```
DeepGuard/
├── app.py                  # Streamlit interface and forensic dashboard
├── run_deepguard.py        # One-command setup and launch script
├── evaluate.py             # Dataset evaluation pipeline
├── check_env.py            # Environment verification
├── requirements.txt        # Python dependencies
├── core/
│   ├── detector.py         # Single-model inference wrapper
│   ├── ensemble.py         # Multi-model ensemble and verdict resolution
│   ├── explainer.py        # GradCAM heatmap generation
│   ├── model_registry.py   # Model metadata, benchmarks, and weight paths
│   ├── preprocessor.py     # Frame sampling and MTCNN face detection
│   ├── reporter.py         # LLM report generation via Ollama
│   └── aggregator.py       # Result aggregation utilities
├── models/
│   ├── download_all.py     # Automated weight downloader
│   └── *.pth               # Model weights (not tracked, run download_all.py)
├── utils/
│   └── visualizer.py       # Side-by-side comparison image builder
├── tests/                  # Pytest test suite
├── docs/                   # Architecture diagrams and guides
└── results/                # Evaluation outputs
```

---

## License

This project was developed for academic purposes at FCIT, International Islamic University, Islamabad.

---
