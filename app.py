"""Streamlit interface for DeepGuard v2.0 multi-model deepfake detection."""

from __future__ import annotations

import html
import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


load_dotenv()

LOGGER = logging.getLogger("deepguard.app")
ROOT_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def configure_logging() -> None:
    """Configure shared application logging."""
    log_level = os.getenv("DEEPGUARD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()


try:
    import numpy as np
    import torch

    from core.ensemble import DeepfakeEnsemble
    from core.explainer import GradCAMExplainer
    from core.model_registry import MODEL_REGISTRY, get_available_models, get_model_config
    from core.preprocessor import VideoPreprocessor
    from core.reporter import LLMReporter
    from utils.visualizer import build_side_by_side
except Exception as import_error:
    st.error(
        "DeepGuard could not import one or more required modules. "
        "Install dependencies with `pip install -r requirements.txt` and run `python check_env.py`."
    )
    st.code(str(import_error))
    st.stop()


CSS = """
<style>
:root {
    --bg: #08111f;
    --bg-2: #101b31;
    --card: rgba(18, 28, 47, 0.84);
    --card-2: rgba(11, 18, 33, 0.92);
    --card-3: rgba(20, 31, 52, 0.72);
    --text: #edf4ff;
    --muted: #9eb2d2;
    --muted-2: #6f85a7;
    --real: #12f0a6;
    --fake: #ff5f6d;
    --warn: #ffd166;
    --accent: #58a6ff;
    --teal: #3de5d3;
    --gold: #ffd700;
    --border: rgba(255, 255, 255, 0.09);
    --border-strong: rgba(255, 255, 255, 0.14);
    --shadow: 0 28px 70px rgba(0, 0, 0, 0.28);
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(18, 240, 166, 0.13), transparent 22%),
        radial-gradient(circle at 88% 10%, rgba(88, 166, 255, 0.12), transparent 23%),
        radial-gradient(circle at 85% 88%, rgba(255, 95, 109, 0.11), transparent 24%),
        linear-gradient(180deg, #08111f 0%, #091321 35%, #0b1629 100%);
    color: var(--text);
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 34px 34px;
}

.main .block-container {
    max-width: 1440px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(88, 166, 255, 0.12), transparent 24%),
        linear-gradient(180deg, rgba(8, 13, 24, 0.98), rgba(10, 17, 31, 0.98));
    border-right: 1px solid var(--border-strong);
}

.dg-card {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)),
        var(--card);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.2rem 1.25rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
}

.dg-banner {
    border-radius: 26px;
    padding: 1.5rem 1.65rem;
    margin-bottom: 1.25rem;
    border: 2px solid transparent;
    animation: pulse-border 2.4s infinite;
    position: relative;
    overflow: hidden;
}

.dg-banner::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255,255,255,0.13), transparent 32%, transparent 68%, rgba(255,255,255,0.08));
    opacity: 0.45;
    pointer-events: none;
}

.dg-banner.fake {
    background: linear-gradient(135deg, rgba(255, 95, 109, 0.22), rgba(97, 13, 30, 0.18));
    border-color: rgba(255, 95, 109, 0.92);
    box-shadow: 0 0 34px rgba(255, 95, 109, 0.2);
}

.dg-banner.real {
    background: linear-gradient(135deg, rgba(18, 240, 166, 0.22), rgba(4, 73, 61, 0.18));
    border-color: rgba(18, 240, 166, 0.92);
    box-shadow: 0 0 34px rgba(18, 240, 166, 0.18);
}

.dg-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.42rem 0.82rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: var(--teal);
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
}

.dg-hero {
    position: relative;
    overflow: hidden;
    padding: 1.7rem 1.8rem;
    border-radius: 30px;
    border: 1px solid var(--border-strong);
    background:
        radial-gradient(circle at 0% 0%, rgba(88, 166, 255, 0.24), transparent 32%),
        radial-gradient(circle at 100% 100%, rgba(18, 240, 166, 0.16), transparent 28%),
        linear-gradient(145deg, rgba(14, 24, 43, 0.96), rgba(8, 14, 27, 0.96));
    box-shadow: 0 34px 80px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.2rem;
}

.dg-hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.95fr);
    gap: 1.15rem;
    align-items: stretch;
}

.dg-hero h1 {
    margin: 0.7rem 0 0.55rem 0;
    font-size: clamp(2.3rem, 4.2vw, 4rem);
    line-height: 1.02;
    letter-spacing: -0.04em;
}

.dg-hero p {
    margin: 0;
    color: var(--muted);
    font-size: 1.02rem;
    line-height: 1.7;
    max-width: 60ch;
}

.dg-pill-row,
.dg-model-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.dg-pill,
.dg-model-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04);
    color: var(--text);
    font-size: 0.9rem;
}

.dg-pill.real { color: var(--real); }
.dg-pill.fake { color: var(--fake); }
.dg-pill.warn { color: var(--warn); }
.dg-pill.accent { color: var(--accent); }

.dg-hero-side {
    padding: 1rem 1.05rem;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
}

.dg-hero-side h3,
.dg-panel h3 {
    margin: 0 0 0.8rem 0;
    font-size: 1rem;
    letter-spacing: 0.02em;
}

.dg-hero-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
}

.dg-hero-stat,
.dg-metric-card,
.dg-mini-card {
    padding: 1rem 1rem 0.95rem;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(8, 15, 28, 0.44);
}

.dg-hero-stat .value,
.dg-metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
}

.dg-hero-stat .label,
.dg-metric-label {
    color: var(--muted);
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.dg-hero-stat .sub,
.dg-metric-sub {
    color: var(--muted-2);
    font-size: 0.88rem;
    margin-top: 0.38rem;
    line-height: 1.45;
}

.dg-section-head {
    margin: 0.3rem 0 0.95rem 0;
}

.dg-eyebrow {
    color: var(--teal);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.38rem;
}

.dg-section-head h2 {
    margin: 0;
    font-size: 1.45rem;
    letter-spacing: -0.03em;
}

.dg-section-head p {
    margin: 0.4rem 0 0 0;
    color: var(--muted);
    max-width: 70ch;
    line-height: 1.65;
}

.dg-metric-card.real {
    border-color: rgba(18, 240, 166, 0.28);
    box-shadow: inset 0 0 0 1px rgba(18, 240, 166, 0.05);
}

.dg-metric-card.fake {
    border-color: rgba(255, 95, 109, 0.28);
    box-shadow: inset 0 0 0 1px rgba(255, 95, 109, 0.05);
}

.dg-metric-card.accent {
    border-color: rgba(88, 166, 255, 0.28);
    box-shadow: inset 0 0 0 1px rgba(88, 166, 255, 0.05);
}

.dg-metric-card.warn {
    border-color: rgba(255, 209, 102, 0.28);
    box-shadow: inset 0 0 0 1px rgba(255, 209, 102, 0.05);
}

.dg-panel {
    padding: 1.2rem 1.2rem 1.05rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        var(--card-2);
    box-shadow: var(--shadow);
}

.dg-panel p {
    color: var(--muted);
    line-height: 1.65;
}

.dg-note {
    border-left: 3px solid var(--accent);
    padding: 0.15rem 0 0.15rem 0.85rem;
    color: var(--muted);
    margin: 0.75rem 0 0 0;
}

.dg-table-wrap {
    overflow: hidden;
    border-radius: 22px;
    border: 1px solid var(--border);
    background: rgba(8, 14, 26, 0.52);
    box-shadow: var(--shadow);
}

.dg-table {
    width: 100%;
    border-collapse: collapse;
}

.dg-table thead th {
    text-align: left;
    padding: 0.92rem 1rem;
    background: rgba(255,255,255,0.04);
    color: var(--muted);
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}

.dg-table tbody td {
    padding: 0.95rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    vertical-align: middle;
}

.dg-table tbody tr:last-child td {
    border-bottom: none;
}

.dg-table tbody tr.winner {
    background: linear-gradient(90deg, rgba(88, 166, 255, 0.1), rgba(18, 240, 166, 0.04));
}

.dg-verdict-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 74px;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    border: 1px solid transparent;
}

.dg-verdict-chip.fake {
    color: #ffd7db;
    background: rgba(255, 95, 109, 0.14);
    border-color: rgba(255, 95, 109, 0.25);
}

.dg-verdict-chip.real {
    color: #d5fff0;
    background: rgba(18, 240, 166, 0.14);
    border-color: rgba(18, 240, 166, 0.25);
}

.dg-winner-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.6rem;
    border-radius: 999px;
    color: var(--gold);
    background: rgba(255, 215, 0, 0.1);
    border: 1px solid rgba(255, 215, 0, 0.24);
    font-size: 0.78rem;
    font-weight: 700;
}

.dg-compare-stack {
    display: grid;
    gap: 0.95rem;
}

.dg-compare-row {
    padding: 1.08rem 1.1rem;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        rgba(10, 18, 32, 0.76);
    box-shadow: var(--shadow);
}

.dg-compare-row.winner {
    border-color: rgba(88, 166, 255, 0.34);
    background:
        radial-gradient(circle at top right, rgba(18,240,166,0.12), transparent 30%),
        linear-gradient(180deg, rgba(88,166,255,0.08), rgba(255,255,255,0.01)),
        rgba(10, 18, 32, 0.9);
}

.dg-compare-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.95rem;
}

.dg-compare-title {
    margin: 0;
    font-size: 1.08rem;
    letter-spacing: -0.02em;
}

.dg-compare-sub {
    margin-top: 0.38rem;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.55;
}

.dg-compare-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}

.dg-compare-cell {
    padding: 0.85rem 0.88rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}

.dg-compare-key {
    color: var(--muted);
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
}

.dg-compare-value {
    margin-top: 0.42rem;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.35;
}

.dg-compare-value.small {
    font-size: 0.95rem;
    font-weight: 600;
}

.dg-steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 0.8rem;
}

.dg-step {
    padding: 1rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}

.dg-step-index {
    display: inline-flex;
    width: 32px;
    height: 32px;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    margin-bottom: 0.75rem;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    color: #07111f;
    font-weight: 800;
}

.dg-step h4 {
    margin: 0 0 0.3rem 0;
}

.dg-step p {
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
}

.dg-status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.95rem;
}

.dg-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.dg-dot.ok { background: var(--real); box-shadow: 0 0 12px rgba(18,240,166,0.6); }
.dg-dot.bad { background: var(--fake); box-shadow: 0 0 12px rgba(255,95,109,0.6); }
.dg-dot.warn { background: var(--warn); box-shadow: 0 0 12px rgba(255,209,102,0.6); }

.dg-report {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        var(--card-2);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.1rem;
    box-shadow: var(--shadow);
}

.dg-report pre {
    margin: 0;
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    white-space: pre-wrap;
}

.dg-report-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.95fr);
    gap: 1rem;
    align-items: start;
}

.dg-report-stack {
    display: grid;
    gap: 1rem;
}

.dg-report-hero-card,
.dg-report-section {
    padding: 1.2rem 1.2rem 1.1rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        var(--card-2);
    box-shadow: var(--shadow);
}

.dg-report-hero-card {
    background:
        radial-gradient(circle at top right, rgba(88,166,255,0.12), transparent 28%),
        radial-gradient(circle at bottom left, rgba(18,240,166,0.10), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        var(--card-2);
}

.dg-report-hero-card h3,
.dg-report-section h3 {
    margin: 0;
    font-size: 1.08rem;
    letter-spacing: -0.02em;
}

.dg-report-kicker {
    color: var(--teal);
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.dg-report-headline {
    margin: 0.45rem 0 0 0;
    font-size: 1.42rem;
    line-height: 1.2;
    letter-spacing: -0.03em;
}

.dg-report-copy {
    color: var(--muted);
    line-height: 1.7;
    margin-top: 0.8rem;
}

.dg-report-copy strong {
    color: var(--text);
}

.dg-report-list {
    margin: 0.85rem 0 0 0;
    padding-left: 1.15rem;
    color: var(--muted);
}

.dg-report-list li {
    margin-bottom: 0.58rem;
    line-height: 1.55;
}

.dg-spotlight-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 0.9rem;
}

.dg-spotlight {
    padding: 0.9rem 0.92rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}

.dg-spotlight-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.dg-spotlight-value {
    margin-top: 0.38rem;
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.3;
}

.dg-flow-grid {
    display: grid;
    gap: 0.72rem;
    margin-top: 0.9rem;
}

.dg-flow-step {
    padding: 0.88rem 0.92rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}

.dg-flow-step strong {
    display: block;
    margin-bottom: 0.2rem;
    color: var(--text);
}

.dg-flow-step span {
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.94rem;
}

.dg-model-meta {
    color: var(--muted);
    font-size: 0.92rem;
    margin-top: -0.4rem;
    margin-bottom: 0.8rem;
}

.dg-small {
    color: var(--muted);
    font-size: 0.95rem;
}

div[data-baseweb="tab-list"] {
    gap: 0.7rem;
    background: transparent;
    margin: 0.35rem 0 1rem 0;
}

button[data-baseweb="tab"] {
    height: 48px;
    padding: 0 1rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: var(--muted);
    font-weight: 700;
}

button[aria-selected="true"][data-baseweb="tab"] {
    color: var(--text);
    background: linear-gradient(135deg, rgba(88, 166, 255, 0.17), rgba(18, 240, 166, 0.14));
    border-color: rgba(88, 166, 255, 0.36);
}

div[data-testid="stFileUploader"] {
    padding: 0.2rem 0 0.5rem 0;
}

div[data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(88, 166, 255, 0.35);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    padding: 1.25rem 1rem;
}

div[data-testid="stVideo"] {
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

div[data-testid="stImage"] img {
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 18px 40px rgba(0,0,0,0.24);
}

div[data-testid="stAlert"] {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

button[kind="primary"] {
    border-radius: 999px;
    border: none;
    background: linear-gradient(135deg, #1f7cff, #12f0a6);
    color: #07111f;
    font-weight: 800;
    letter-spacing: 0.02em;
    box-shadow: 0 18px 40px rgba(18, 240, 166, 0.18);
}

button[kind="secondary"] {
    border-radius: 999px;
}

@media (max-width: 980px) {
    .dg-hero-grid,
    .dg-steps,
    .dg-hero-stats,
    .dg-report-grid {
        grid-template-columns: 1fr;
    }

    .dg-compare-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .dg-spotlight-grid {
        grid-template-columns: 1fr;
    }

    .main .block-container {
        padding-top: 1rem;
    }
}

@media (max-width: 640px) {
    .dg-compare-top,
    .dg-compare-grid {
        grid-template-columns: 1fr;
    }

    .dg-compare-top {
        display: grid;
    }
}

@keyframes pulse-border {
    0% { box-shadow: 0 0 0 rgba(0, 0, 0, 0.0); }
    50% { box-shadow: 0 0 32px rgba(255,255,255,0.05); }
    100% { box-shadow: 0 0 0 rgba(0, 0, 0, 0.0); }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def resolve_device(selection: str) -> str:
    """Convert the sidebar selection to a torch device string."""
    if selection.startswith("CUDA") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_uploaded_video(uploaded_file) -> Path:
    """Persist the uploaded file to a temporary location for OpenCV access."""
    temp_dir = ROOT_DIR / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def load_reporter(ollama_url: str, model: str) -> LLMReporter:
    """Build a fresh reporter so status stays current."""
    return LLMReporter(ollama_url=ollama_url, model=model)


@st.cache_resource(show_spinner=False)
def load_ensemble(selected_model_keys: tuple[str, ...], device: str, use_weighted_voting: bool) -> DeepfakeEnsemble:
    """Cache an ensemble for the selected model set, device, and voting mode."""
    return DeepfakeEnsemble(
        selected_model_keys=list(selected_model_keys),
        device=device,
        use_weighted_voting=use_weighted_voting,
    )


def format_status(label: str, state: str, text: str) -> str:
    """Return a small colored status badge for the sidebar."""
    return f"<div class='dg-status'><span class='dg-dot {state}'></span><strong>{label}:</strong> {text}</div>"


def sorted_model_keys_by_auc() -> list[str]:
    """Return registry keys sorted by benchmark score descending."""
    return sorted(MODEL_REGISTRY, key=lambda key: MODEL_REGISTRY[key]["auc"], reverse=True)


def build_model_checkbox_label(model_key: str, available: bool) -> str:
    """Create the sidebar checkbox label for one model."""
    config = get_model_config(model_key)
    recommended = " [RECOMMENDED]" if config["recommended"] else ""
    if not available:
        recommended = " ⚠️ Run download_all.py"
    if model_key == "mesonet4":
        recommended = " ⚠️ Lower accuracy"
    return f"{config['display_name']}{recommended}"


def render_app_hero(selected_model_keys: list[str], available_model_count: int, ollama_online: bool) -> None:
    """Render the top-of-page DeepGuard hero shell."""
    selected_models = "".join(
        f"<span class='dg-model-pill'>{html.escape(get_model_config(key)['display_name'])}</span>"
        for key in selected_model_keys[:4]
    )
    if len(selected_model_keys) > 4:
        selected_models += f"<span class='dg-model-pill'>+{len(selected_model_keys) - 4} more</span>"
    if not selected_models:
        selected_models = "<span class='dg-model-pill'>Select at least one model to begin</span>"

    ollama_label = "Ollama Connected" if ollama_online else "Template Report Mode"
    ollama_class = "real" if ollama_online else "warn"
    st.markdown(
        f"""
        <div class='dg-hero'>
            <div class='dg-hero-grid'>
                <div>
                    <div class='dg-kicker'>Deepfake Forensics Workspace</div>
                    <h1>DeepGuard</h1>
                    <p>
                        Multi-model video screening with face-level inspection, visual explanations,
                        and a polished forensic workflow built for fast authenticity review.
                    </p>
                    <div class='dg-pill-row'>
                        <span class='dg-pill real'>{available_model_count} models ready</span>
                        <span class='dg-pill'>{len(selected_model_keys)} selected for this run</span>
                        <span class='dg-pill {ollama_class}'>{ollama_label}</span>
                    </div>
                    <div class='dg-model-pill-row'>{selected_models}</div>
                </div>
                <div class='dg-hero-side'>
                    <h3>Mission Snapshot</h3>
                    <div class='dg-hero-stats'>
                        <div class='dg-hero-stat'>
                            <div class='label'>Pipeline</div>
                            <div class='value'>Detect</div>
                            <div class='sub'>Frame sampling, face crops, classifier passes.</div>
                        </div>
                        <div class='dg-hero-stat'>
                            <div class='label'>Evidence</div>
                            <div class='value'>Explain</div>
                            <div class='sub'>GradCAM highlights the most suspicious facial regions.</div>
                        </div>
                        <div class='dg-hero-stat'>
                            <div class='label'>Delivery</div>
                            <div class='value'>Report</div>
                            <div class='sub'>Clear verdict summary for non-technical reviewers.</div>
                        </div>
                        <div class='dg-hero-stat'>
                            <div class='label'>Interface</div>
                            <div class='value'>Refined</div>
                            <div class='sub'>Presentation upgraded without changing model behavior.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent section heading block."""
    st.markdown(
        f"""
        <div class='dg-section-head'>
            <div class='dg-eyebrow'>{html.escape(eyebrow)}</div>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[dict]) -> None:
    """Render the main result metrics as styled cards."""
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        tone = card.get("tone", "accent")
        subtitle = card.get("subtitle", "")
        column.markdown(
            f"""
            <div class='dg-metric-card {tone}'>
                <div class='dg-metric-label'>{html.escape(card['label'])}</div>
                <div class='dg-metric-value'>{html.escape(card['value'])}</div>
                <div class='dg-metric-sub'>{html.escape(subtitle)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def html_list(items: list[str], class_name: str = "dg-report-list") -> str:
    """Render a safe HTML list from plain strings."""
    entries = "".join(f"<li>{html.escape(item)}</li>" for item in items if item)
    return f"<ul class='{class_name}'>{entries}</ul>" if entries else ""


def render_authenticity_report(
    report_payload: dict,
    analysis_result: dict,
    winning_config: dict,
    gradcam_results: list[dict],
    source_filename: str,
) -> None:
    """Render the structured DeepGuard report with project context and export actions."""
    verdict = analysis_result["final_verdict"]
    verdict_class = "fake" if verdict == "FAKE" else "real"
    report_mode_label = "Local LLaMA narrative" if report_payload["source"] == "llm" else "Template fallback narrative"
    report_mode_tone = "accent" if report_payload["source"] == "llm" else "warn"
    suspicious_regions = sorted(
        {item["suspicious_region"] for item in gradcam_results if item.get("suspicious_region")}
    )
    regions_summary = ", ".join(suspicious_regions) if suspicious_regions else "not available for this run"
    flow_steps = [
        ("Detect", f"{analysis_result['frame_count_extracted']} sampled frames produced {analysis_result['face_count_detected']} face crops."),
        ("Compare", f"{analysis_result['num_models_used']} selected model(s) reviewed those face crops."),
        ("Resolve", analysis_result["resolution_explanation"]),
        (
            "Explain",
            (
                f"GradCAM focused on {winning_config['display_name']} and highlighted {regions_summary}."
                if gradcam_results
                else "GradCAM visual evidence was not generated in this run."
            ),
        ),
    ]
    flow_html = "".join(
        f"<div class='dg-flow-step'><strong>{html.escape(title)}</strong><span>{html.escape(description)}</span></div>"
        for title, description in flow_steps
    )
    markdown_filename = f"{Path(source_filename).stem}_deepguard_report.md"

    st.markdown(
        f"""
        <div class='dg-report-grid'>
            <div class='dg-report-stack'>
                <div class='dg-report-hero-card'>
                    <div class='dg-report-kicker'>Narrative Briefing</div>
                    <div class='dg-pill-row'>
                        <span class='dg-pill {verdict_class}'>{html.escape(verdict)}</span>
                        <span class='dg-pill'>{html.escape(report_payload['report_engine'])}</span>
                        <span class='dg-pill {report_mode_tone}'>{html.escape(report_mode_label)}</span>
                    </div>
                    <h3 class='dg-report-headline'>{html.escape(report_payload['headline'])}</h3>
                    <div class='dg-report-copy'><strong>Executive summary:</strong> {html.escape(report_payload['executive_summary'])}</div>
                    <div class='dg-note'>{html.escape(report_payload['plain_language_brief'])}</div>
                </div>
                <div class='dg-report-section'>
                    <h3>Technical Findings</h3>
                    {html_list(report_payload['technical_findings'])}
                </div>
                <div class='dg-report-section'>
                    <h3>Ensemble Analysis</h3>
                    <div class='dg-report-copy'>{html.escape(report_payload['ensemble_analysis'])}</div>
                </div>
                <div class='dg-report-section'>
                    <h3>Confidence Assessment</h3>
                    <div class='dg-report-copy'>{html.escape(report_payload['confidence_assessment'])}</div>
                </div>
                <div class='dg-report-section'>
                    <h3>Recommended Actions</h3>
                    {html_list(report_payload['recommended_actions'])}
                </div>
            </div>
            <div class='dg-report-stack'>
                <div class='dg-report-section'>
                    <h3>How DeepGuard Reached This Result</h3>
                    <div class='dg-report-copy'>{html.escape(report_payload['project_overview'])}</div>
                    <div class='dg-flow-grid'>{flow_html}</div>
                </div>
                <div class='dg-report-section'>
                    <h3>Evidence Highlights</h3>
                    {html_list(report_payload['evidence_highlights'])}
                    <div class='dg-note'>Determining model: <strong>{html.escape(winning_config['display_name'])}</strong></div>
                    <div class='dg-note'>Suspicious regions: <strong>{html.escape(regions_summary)}</strong></div>
                </div>
                <div class='dg-report-section'>
                    <h3>Run Snapshot</h3>
                    <div class='dg-spotlight-grid'>
                        <div class='dg-spotlight'>
                            <div class='dg-spotlight-label'>Fake Probability</div>
                            <div class='dg-spotlight-value'>{analysis_result['final_confidence_pct']:.1f}%</div>
                        </div>
                        <div class='dg-spotlight'>
                            <div class='dg-spotlight-label'>Determining Model</div>
                            <div class='dg-spotlight-value'>{html.escape(winning_config['display_name'])}</div>
                        </div>
                        <div class='dg-spotlight'>
                            <div class='dg-spotlight-label'>Models Used</div>
                            <div class='dg-spotlight-value'>{analysis_result['num_models_used']}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    download_col, preview_col = st.columns([0.78, 1.22], gap="large")
    with download_col:
        st.download_button(
            "Download Markdown Report",
            data=report_payload["markdown"],
            file_name=markdown_filename,
            mime="text/markdown",
            use_container_width=True,
        )
    with preview_col:
        with st.expander("Copy-ready plain text report", expanded=False):
            st.code(report_payload["plain_text"])


def render_comparison_table(analysis_result: dict) -> None:
    """Render the per-model comparison as a native Streamlit table."""
    comparison_df = build_comparison_dataframe(analysis_result)
    st.dataframe(comparison_df, width="stretch", hide_index=True)


def render_selected_run_profile(
    selected_model_keys: list[str],
    frame_skip: int,
    max_frames: int,
    device_choice: str,
    show_gradcam: bool,
    generate_llm_report: bool,
    use_weighted_voting: bool,
) -> None:
    """Render the current analysis configuration summary."""
    selected_models = "".join(
        f"<span class='dg-model-pill'>{html.escape(get_model_config(key)['display_name'])}</span>"
        for key in selected_model_keys
    )
    if not selected_models:
        selected_models = "<span class='dg-model-pill'>No models selected yet</span>"

    voting_label = "Benchmark-weighted voting" if use_weighted_voting else "Majority vote + tiebreak"
    gradcam_label = "Enabled" if show_gradcam else "Disabled"
    report_label = "Enabled" if generate_llm_report else "Disabled"
    st.markdown(
        f"""
        <div class='dg-panel'>
            <h3>Current Analysis Profile</h3>
            <p>Review the active runtime settings before launching a scan.</p>
            <div class='dg-model-pill-row'>{selected_models}</div>
            <div class='dg-note'>
                Frame skip: <strong>{frame_skip}</strong> · Max frames: <strong>{max_frames}</strong> ·
                Device: <strong>{html.escape(device_choice)}</strong>
            </div>
            <div class='dg-note'>
                GradCAM: <strong>{gradcam_label}</strong> · LLM report: <strong>{report_label}</strong> ·
                Resolver: <strong>{voting_label}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_comparison_dataframe(analysis_result: dict) -> pd.DataFrame:
    """Create the comparison table dataframe sorted by benchmark score descending."""
    rows = []
    for row in analysis_result["comparison_table"]:
        rows.append(
            {
                "Role": "Determining model" if row["is_winner"] else "Supporting model",
                "Model": row["display_name"],
                "Verdict": row["verdict"],
                "Fake Probability": f"{row['fake_prob_pct']:.1f}%",
                "Flagged Frames": f"{row['fake_frames']}/{row['total_frames']}",
                "Benchmark Score": row["score_display"],
                "Speed": row["speed"],
            }
        )
    return pd.DataFrame(rows)


def build_timeline_dataframe(analysis_result: dict) -> pd.DataFrame:
    """Build a model-keyed fake-probability timeline dataframe for plotting."""
    timeline_data = {}
    max_length = 0
    for model_key in sorted_model_keys_by_auc():
        if model_key not in analysis_result["fake_prob_timelines"]:
            continue
        label = get_model_config(model_key)["display_name"]
        timeline = analysis_result["fake_prob_timelines"][model_key]
        timeline_data[label] = timeline
        max_length = max(max_length, len(timeline))

    dataframe = pd.DataFrame({"frame_number": list(range(max_length))})
    for label, timeline in timeline_data.items():
        padded = timeline + [np.nan] * (max_length - len(timeline))
        dataframe[label] = padded
    return dataframe


def plot_timeline_chart(analysis_result: dict) -> plt.Figure:
    """Render the multi-model fake-probability timeline with a 0.5 decision boundary."""
    timeline_df = build_timeline_dataframe(analysis_result)
    fig, ax = plt.subplots(figsize=(9.4, 4.35))
    palette = ["#12f0a6", "#58a6ff", "#ffd166", "#ff7b72", "#7c9cff", "#c77dff"]
    ax.set_facecolor("#101827")
    fig.patch.set_facecolor("#101827")
    fig.patch.set_alpha(0.0)

    color_index = 0
    for column in timeline_df.columns:
        if column == "frame_number":
            continue
        color = palette[color_index % len(palette)]
        color_index += 1
        series = pd.to_numeric(timeline_df[column], errors="coerce")
        ax.plot(
            timeline_df["frame_number"],
            series,
            marker="o",
            linewidth=2.5,
            markersize=5.5,
            color=color,
            label=column,
        )
        ax.fill_between(timeline_df["frame_number"], series.fillna(np.nan), 0, alpha=0.08, color=color)
    ax.axhline(0.5, linestyle="--", color="#ffcc33", linewidth=1.5, label="Decision boundary")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Frame number", color="#c7d5ea")
    ax.set_ylabel("Fake probability", color="#c7d5ea")
    ax.set_title("Fake Probability Timeline", color="#edf4ff", pad=12)
    ax.tick_params(colors="#9eb2d2")
    ax.grid(alpha=0.18, color="#9eb2d2")
    for spine in ax.spines.values():
        spine.set_color("#223455")
    legend = ax.legend(loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("#0d1526")
    legend.get_frame().set_edgecolor("#223455")
    for text in legend.get_texts():
        text.set_color("#edf4ff")
    fig.tight_layout()
    return fig


def analyze_video(
    video_path: Path,
    filename: str,
    selected_model_keys: list[str],
    frame_skip: int,
    max_frames: int,
    device: str,
    show_gradcam: bool,
    generate_report: bool,
    use_weighted_voting: bool,
) -> dict:
    """Run the full DeepGuard v2.0 pipeline for one uploaded video."""
    progress_bar = st.progress(0)
    status_box = st.empty()

    def update(step: int, message: str) -> None:
        progress_bar.progress(step)
        status_box.info(message)

    def progress_callback(model_key: str, completed: int, total: int) -> None:
        model_name = get_model_config(model_key)["display_name"]
        step = 20 + int((completed / max(total, 1)) * 55)
        update(step, f"Running {model_name}...")

    try:
        update(5, "Extracting frames...")
        preprocessor = VideoPreprocessor(device=device, frame_skip=frame_skip, max_frames=max_frames)
        frames = preprocessor.extract_frames(str(video_path))

        update(15, "Detecting faces...")
        face_data = preprocessor.detect_and_crop_faces(frames)

        available_models = get_available_models()
        available_selected = [key for key in selected_model_keys if key in available_models]
        missing_selected = [key for key in selected_model_keys if key not in available_models]
        if not available_selected:
            raise FileNotFoundError("All selected models are missing weight files.")

        ensemble = load_ensemble(tuple(available_selected), preprocessor.device, use_weighted_voting)
        analysis_result = ensemble.run(face_data=face_data, preprocessor=preprocessor, progress_callback=progress_callback)
        analysis_result["frame_count_extracted"] = len(frames)
        analysis_result["face_count_detected"] = len(face_data)
        analysis_result["missing_selected_models"] = missing_selected

        winning_model_key = analysis_result["winning_model"]
        winning_detector = ensemble.detectors[winning_model_key]
        winning_frame_results = analysis_result["per_model_frame_results"][winning_model_key]

        gradcam_results: list[dict] = []
        if show_gradcam:
            update(80, f"Generating GradCAM ({winning_detector.config['display_name']})...")
            explainer = GradCAMExplainer(winning_detector, winning_detector.device)
            gradcam_results = explainer.process_top_frames(
                top_frame_indices=analysis_result["top_suspicious_frames"],
                face_data=face_data,
            )
            probability_lookup = {result["frame_idx"]: result["fake_prob"] for result in winning_frame_results}
            for item in gradcam_results:
                item["fake_prob"] = probability_lookup.get(item["frame_idx"], 0.0)
                item["comparison_image"] = build_side_by_side(item["resized_face_rgb"], item["heatmap_overlay"])

        report_payload: dict | None = None
        report_text = ""
        reporter = load_reporter(OLLAMA_URL, OLLAMA_MODEL) if generate_report else None
        if generate_report:
            update(95, "Writing report...")
            report_payload = reporter.generate_report(analysis_result, gradcam_results, filename)
            report_text = report_payload["plain_text"]

        warnings = list(preprocessor.runtime_warnings)
        warnings.extend(ensemble.runtime_warnings)
        if missing_selected:
            missing_names = ", ".join(get_model_config(key)["display_name"] for key in missing_selected)
            warnings.append(f"Missing weight files for selected models: {missing_names}")
        if reporter and reporter.connection_error:
            warnings.append(reporter.connection_error)

        return {
            "analysis_result": analysis_result,
            "gradcam_results": gradcam_results,
            "report_payload": report_payload,
            "report_text": report_text,
            "source_filename": filename,
            "warnings": warnings,
        }
    finally:
        progress_bar.empty()
        status_box.empty()


available_models = get_available_models()
reporter_status = load_reporter(OLLAMA_URL, OLLAMA_MODEL)

st.sidebar.markdown(
    """
    <div class='dg-card'>
        <div class='dg-kicker'>Deepfake Detection Console</div>
        <h2 style='margin:0.9rem 0 0.4rem 0;'>🛡️ DeepGuard v2.0</h2>
        <p class='dg-small' style='margin:0;'>Explainable video screening with a refined forensic dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Select Models")

selected_model_keys: list[str] = []
for model_key in sorted_model_keys_by_auc():
    config = get_model_config(model_key)
    is_available = model_key in available_models
    default_value = bool(config["recommended"] and is_available and model_key != "efficientnet_b7")
    selected = st.sidebar.checkbox(
        build_model_checkbox_label(model_key, is_available),
        value=default_value,
        disabled=not is_available,
        key=f"model_select_{model_key}",
    )
    st.sidebar.markdown(
        f"<div class='dg-model-meta'>{config['subtitle']} · {config['score_display']} · {config['speed']}</div>",
        unsafe_allow_html=True,
    )
    if selected and is_available:
        selected_model_keys.append(model_key)

use_weighted_voting = st.sidebar.toggle("Advanced: benchmark-weighted soft voting", value=False)
if len(selected_model_keys) > 1:
    if use_weighted_voting:
        st.sidebar.info(
            "ℹ️ Resolution: benchmark-weighted voting\n\nWinner chosen by published benchmark score, not by per-video confidence score."
        )
    else:
        st.sidebar.info(
            "ℹ️ Resolution: Majority vote + benchmark tiebreaking\n\nWinner chosen by published benchmark score, not by per-video confidence score."
        )

st.sidebar.markdown("━━━━━━━━━━━━━━━━━")
st.sidebar.markdown("### Settings")
frame_skip = st.sidebar.slider("Frame Skip", min_value=5, max_value=30, value=10)
max_frames = st.sidebar.slider("Max Frames", min_value=10, max_value=50, value=30)
device_choice = st.sidebar.selectbox("Device", ["CPU", "CUDA"])
show_gradcam = st.sidebar.checkbox("Show GradCAM", value=True)
generate_llm_report = st.sidebar.checkbox("LLM Report", value=True)

st.sidebar.markdown("### System Status")
st.sidebar.markdown(
    format_status("Models ready", "ok" if available_models else "warn", f"{len(available_models)} / {len(MODEL_REGISTRY)}"),
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    format_status("Ollama", "ok" if reporter_status.online else "bad", "Connected" if reporter_status.online else "Offline"),
    unsafe_allow_html=True,
)

run_disabled = len(selected_model_keys) == 0
if run_disabled:
    st.sidebar.error("Select at least one model.")

render_app_hero(selected_model_keys, len(available_models), reporter_status.online)

analyze_tab, models_tab, how_tab, about_tab = st.tabs(["Analyze Video", "Model Information", "How It Works", "About"])

with analyze_tab:
    render_section_header(
        "Analysis Suite",
        "Analyze Video",
        "Upload a suspicious clip, launch the deepfake pipeline, and review the verdict, evidence, and written authenticity summary in one place.",
    )

    upload_col, profile_col = st.columns([1.45, 1.0], gap="large")
    with upload_col:
        st.markdown(
            """
            <div class='dg-panel'>
                <h3>Upload Evidence</h3>
                <p>Supported formats: MP4, AVI, MOV, and MKV. DeepGuard will sample frames, crop faces, run the selected models, and assemble an explanation-first result view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            st.markdown(
                f"<div class='dg-note'>Loaded file: <strong>{html.escape(uploaded_file.name)}</strong></div>",
                unsafe_allow_html=True,
            )
            st.video(uploaded_file)

    with profile_col:
        render_selected_run_profile(
            selected_model_keys=selected_model_keys,
            frame_skip=frame_skip,
            max_frames=max_frames,
            device_choice=device_choice,
            show_gradcam=show_gradcam,
            generate_llm_report=generate_llm_report,
            use_weighted_voting=use_weighted_voting,
        )
        st.write("")
        st.markdown(
            """
            <div class='dg-panel'>
                <h3>Operator Guidance</h3>
                <p>Use lower frame skip for harder cases, keep GradCAM enabled for explainability, and compare multiple models when a result feels borderline.</p>
                <div class='dg-note'>This refresh changes only presentation. The detection pipeline and model behavior remain exactly the same.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    run_clicked = st.button("Run DeepGuard Analysis", type="primary", width="stretch", disabled=run_disabled)
    if run_clicked and uploaded_file is None:
        st.warning("Upload a video before starting analysis.")

    if run_clicked and uploaded_file is not None:
        temp_video_path = save_uploaded_video(uploaded_file)
        try:
            result_bundle = analyze_video(
                video_path=temp_video_path,
                filename=uploaded_file.name,
                selected_model_keys=selected_model_keys,
                frame_skip=frame_skip,
                max_frames=max_frames,
                device=resolve_device(device_choice),
                show_gradcam=show_gradcam,
                generate_report=generate_llm_report,
                use_weighted_voting=use_weighted_voting,
            )
            st.session_state["deepguard_result"] = result_bundle
        except ValueError as exc:
            if str(exc) == "insufficient_faces":
                st.error("No face detected. Upload a video with a visible human face.")
            else:
                st.error(str(exc))
        except FileNotFoundError as exc:
            st.error(f"{exc} Run `python models/download_all.py` first.")
        except Exception as exc:  # pragma: no cover - UI safeguard
            LOGGER.exception("Unexpected application error")
            st.error(f"DeepGuard encountered an unexpected error: {exc}")
        finally:
            temp_video_path.unlink(missing_ok=True)

    result_bundle = st.session_state.get("deepguard_result")
    if not result_bundle:
        st.write("")
        render_section_header(
            "Workflow",
            "What You’ll See After Analysis",
            "DeepGuard keeps the decision process transparent by surfacing the winning model, frame-level evidence, and a human-readable report together.",
        )
        st.markdown(
            """
            <div class='dg-steps'>
                <div class='dg-step'>
                    <div class='dg-step-index'>1</div>
                    <h4>Verdict</h4>
                    <p>A high-visibility banner announces whether the uploaded video appears authentic or suspicious.</p>
                </div>
                <div class='dg-step'>
                    <div class='dg-step-index'>2</div>
                    <h4>Evidence</h4>
                    <p>Top suspicious faces are paired with GradCAM overlays so the visual rationale stays visible.</p>
                </div>
                <div class='dg-step'>
                    <div class='dg-step-index'>3</div>
                    <h4>Report</h4>
                    <p>The final screen includes a structured narrative report for easy sharing with non-technical reviewers.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result_bundle:
        analysis_result = result_bundle["analysis_result"]
        gradcam_results = result_bundle["gradcam_results"]
        report_payload = result_bundle.get("report_payload")
        report_text = result_bundle["report_text"]
        source_filename = result_bundle.get("source_filename", "deepguard_analysis")

        for warning in result_bundle["warnings"]:
            lowered = warning.lower()
            if "ollama" in lowered or "11434" in lowered:
                st.warning("⚠️ Ollama offline — using template report")
            elif "missing weight files" in lowered:
                st.warning(warning)
            elif "cuda" in lowered:
                st.info(warning)
            else:
                st.warning(warning)

        winning_model_key = analysis_result["winning_model"]
        winning_config = get_model_config(winning_model_key)
        winning_result = analysis_result["per_model_results"][winning_model_key]
        verdict = analysis_result["final_verdict"]
        final_prob_pct = analysis_result["final_confidence_pct"]
        banner_class = "fake" if verdict == "FAKE" else "real"
        banner_text = "⚠️ DEEPFAKE DETECTED" if verdict == "FAKE" else "✅ VIDEO APPEARS AUTHENTIC"
        subtitle = (
            f"Determining model: {winning_config['display_name']} ({winning_config['score_display']}) · "
            f"{analysis_result['resolution_explanation']}"
        )
        st.markdown(
            f"""
            <div class='dg-banner {banner_class}'>
                <div style='font-size: 1.9rem; font-weight: 800;'>{banner_text} — {final_prob_pct:.1f}%</div>
                <div style='font-size: 1.0rem; margin-top: 0.35rem;'>{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis_result["num_models_used"] > 1:
            st.info(f"🔬 {analysis_result['resolution_explanation']}")

        if analysis_result["num_models_used"] == 1 and 0.45 <= analysis_result["final_fake_prob"] <= 0.55:
            st.warning("⚠️ Result is borderline — consider running additional models")

        agree_count = max(analysis_result["fake_vote_count"], analysis_result["real_vote_count"])
        render_metric_cards(
            [
                {
                    "label": "Final Fake Probability",
                    "value": f"{final_prob_pct:.1f}%",
                    "subtitle": "Video-level fake likelihood from the final resolver.",
                    "tone": "fake" if verdict == "FAKE" else "real",
                },
                {
                    "label": "Determining Model",
                    "value": winning_config["display_name"],
                    "subtitle": winning_config["score_display"],
                    "tone": "accent",
                },
                {
                    "label": "Model Agreement",
                    "value": f"{agree_count}/{analysis_result['num_models_used']}",
                    "subtitle": "Models supporting the final verdict.",
                    "tone": "warn",
                },
                {
                    "label": "Face Frames Analyzed",
                    "value": str(winning_result["total_frames"]),
                    "subtitle": f"Detected from {analysis_result['frame_count_extracted']} sampled frames.",
                    "tone": "accent",
                },
                {
                    "label": "Fake Frames",
                    "value": str(winning_result["fake_frame_count"]),
                    "subtitle": f"{analysis_result['face_count_detected']} face crops were available overall.",
                    "tone": "fake" if winning_result["fake_frame_count"] else "real",
                },
            ]
        )

        if analysis_result["num_models_used"] > 1:
            st.write("")
            render_section_header(
                "Ensemble Board",
                "Model Comparison",
                "Each selected detector is shown below with its verdict, video-level fake probability, and published benchmark score.",
            )
            render_comparison_table(analysis_result)
            st.caption(
                f"Winner selected by benchmark-based logic. {analysis_result['resolution_explanation']}"
            )

        st.write("")
        render_section_header(
            "Explainability",
            "GradCAM Heatmaps",
            f"Visual explanations are generated from {winning_config['display_name']}, the highest-ranked model that determined the final verdict.",
        )
        if gradcam_results:
            columns = st.columns(3)
            for index, heatmap in enumerate(gradcam_results[:3]):
                with columns[index]:
                    st.image(
                        heatmap["comparison_image"],
                        caption=f"Frame {heatmap['frame_idx']} — {heatmap['suspicious_region']} — {heatmap.get('fake_prob', 0.0):.1%}",
                        width="stretch",
                    )
        else:
            st.info("GradCAM generation was disabled for this run.")

        st.write("")
        timeline_col, summary_col = st.columns([1.45, 0.95], gap="large")
        with timeline_col:
            render_section_header(
                "Temporal Signal",
                "Confidence Timeline",
                "Frame-by-frame fake probability makes it easier to see whether suspicion is isolated or persistent across the video.",
            )
            st.pyplot(plot_timeline_chart(analysis_result))
            st.caption("Dashed line at 0.5 = decision boundary.")
        with summary_col:
            st.markdown(
                f"""
                <div class='dg-panel'>
                    <h3>Run Summary</h3>
                    <p>{html.escape(analysis_result['resolution_explanation'])}</p>
                    <div class='dg-note'>Verdict: <strong>{verdict}</strong> · Determining model: <strong>{html.escape(winning_config['display_name'])}</strong></div>
                    <div class='dg-note'>Frames extracted: <strong>{analysis_result['frame_count_extracted']}</strong> · Face crops detected: <strong>{analysis_result['face_count_detected']}</strong></div>
                    <div class='dg-note'>Resolution mode: <strong>{'Weighted voting' if use_weighted_voting else 'Majority vote + tiebreak'}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if generate_llm_report:
            st.write("")
            render_section_header(
                "Narrative Report",
                "Authenticity Report",
                "A project-aware summary that explains the verdict, the DeepGuard workflow, and the evidence in a cleaner presentation.",
            )
            if report_payload:
                render_authenticity_report(
                    report_payload=report_payload,
                    analysis_result=analysis_result,
                    winning_config=winning_config,
                    gradcam_results=gradcam_results,
                    source_filename=source_filename,
                )
            else:
                st.markdown(
                    f"<div class='dg-report'><pre>{html.escape(report_text)}</pre></div>",
                    unsafe_allow_html=True,
                )

with models_tab:
    render_section_header(
        "Model Roster",
        "Model Information",
        "DeepGuard presents every available detector with its published benchmark score, runtime profile, and readiness state so the interface stays transparent.",
    )
    model_columns = st.columns(2, gap="large")
    for index, model_key in enumerate(sorted_model_keys_by_auc()):
        config = get_model_config(model_key)
        is_ready = model_key in available_models
        status_text = "Ready" if is_ready else "Download required"
        warning_text = " ⚠️ Lower accuracy — included for comparison" if model_key == "mesonet4" else ""
        model_columns[index % 2].markdown(
            f"""
            <div class='dg-card'>
                <div class='dg-kicker'>{html.escape(status_text)}</div>
                <h3 style='margin:0.9rem 0 0.45rem 0;'>{config['display_name']} <span style='color:#ffd700;'>{config['score_display']}</span></h3>
                <p class='dg-small'>{config['subtitle']} · {config['benchmark_label']} · {config['speed']}</p>
                <p>{config['description']}{warning_text}</p>
                <div class='dg-model-pill-row'>
                    <span class='dg-model-pill'>Input {config['input_size']}×{config['input_size']}</span>
                    <span class='dg-model-pill'>{config['speed']}</span>
                    <span class='dg-model-pill'>{'Recommended' if config['recommended'] else 'Optional'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with how_tab:
    render_section_header(
        "Operational Flow",
        "How It Works",
        "DeepGuard follows a detect → explain → report pattern so the interface tells you not only what happened, but why the system reached that conclusion.",
    )
    st.markdown(
        """
        <div class='dg-steps'>
            <div class='dg-step'>
                <div class='dg-step-index'>1</div>
                <h4>Detect</h4>
                <p>OpenCV samples frames, MTCNN crops faces, and each selected model scores every face crop as real or fake.</p>
            </div>
            <div class='dg-step'>
                <div class='dg-step-index'>2</div>
                <h4>Explain</h4>
                <p>The determining model generates GradCAM overlays for the most suspicious frames so the result remains interpretable.</p>
            </div>
            <div class='dg-step'>
                <div class='dg-step-index'>3</div>
                <h4>Report</h4>
                <p>Ollama or the fallback template produces a concise authenticity report for non-technical reviewers.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_section_header(
        "Resolver Rules",
        "Ensemble Decision Logic",
        "When multiple models are selected, DeepGuard resolves disagreements using benchmark-ranked decision rules instead of arbitrary averaging.",
    )
    st.markdown(
        """
        <div class='dg-table-wrap'>
            <table class='dg-table'>
                <thead>
                    <tr>
                        <th>Case</th>
                        <th>Rule</th>
                        <th>Example</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Single model</strong></td>
                        <td>That model's verdict is used directly.</td>
                        <td>Only Xception selected.</td>
                    </tr>
                    <tr>
                        <td><strong>Unanimous</strong></td>
                        <td>Highest-ranked benchmark model among agreeing voters becomes the reference.</td>
                        <td>All selected models vote FAKE.</td>
                    </tr>
                    <tr>
                        <td><strong>Majority vote</strong></td>
                        <td>Majority side wins; the strongest benchmark model on that side determines the final reference.</td>
                        <td>2 FAKE vs 1 REAL.</td>
                    </tr>
                    <tr>
                        <td><strong>Perfect tie</strong></td>
                        <td>The highest-ranked benchmark model overall breaks the tie.</td>
                        <td>1 FAKE vs 1 REAL.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    flow_col, ref_col = st.columns([1.1, 0.9], gap="large")
    with flow_col:
        render_section_header(
            "Weighted Mode",
            "Benchmark-Weighted Soft Vote",
            "Advanced mode uses the published benchmark score of each model as a reliability weight when combining per-video fake probabilities.",
        )
        st.code(
            "weighted_fake_prob = sum(score_i * video_fake_prob_i) / sum(score_i)\n"
            "final_verdict = 'FAKE' if weighted_fake_prob > 0.5 else 'REAL'",
            language="python",
        )
    with ref_col:
        st.markdown(
            """
            <div class='dg-panel'>
                <h3>References</h3>
                <p>DeepGuard’s current roster and design are informed by well-known academic and benchmark sources used throughout deepfake detection research.</p>
                <div class='dg-note'>Rossler et al. 2019 — FaceForensics++</div>
                <div class='dg-note'>Seferbekov 2020 — DFDC competition leaderboard</div>
                <div class='dg-note'>Wodajo &amp; Atnafu 2021 — transformer-based deepfake detection</div>
                <div class='dg-note'>Afchar et al. 2018 — MesoNet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with about_tab:
    render_section_header(
        "Project Context",
        "About DeepGuard",
        "DeepGuard is an AI-powered deepfake detection and explanation system developed for FCIT, International Islamic University Islamabad.",
    )
    about_col, team_col = st.columns([1.1, 0.9], gap="large")
    with about_col:
        st.markdown(
            """
            <div class='dg-panel'>
                <h3>System Focus</h3>
                <p>DeepGuard combines deepfake detection, visual explainability, and accessible reporting into a single interface designed for classroom demonstration and practical forensic review.</p>
                <div class='dg-note'>Institution: FCIT, International Islamic University Islamabad</div>
                <div class='dg-note'>Program: BSSE F22 B</div>
                <div class='dg-note'>Interface goal: explain model verdicts instead of hiding them.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with team_col:
        st.markdown(
            """
            <div class='dg-panel'>
                <h3>Team</h3>
                <div class='dg-note'>Laiba Shehryar (4523-BSSE-F22-B)</div>
                <div class='dg-note'>Noor Fatima (4531-BSSE-F22-B)</div>
                <div class='dg-note'>Syeda Anooshay Yousuf (4492-BSSE-F22-B)</div>
                <div class='dg-note'>Syeda Ifrah Batool Zaidi (4534-BSSE-F22-B)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
