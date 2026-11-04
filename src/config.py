"""Paths and settings shared across modules."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
TRACE_DIR = ROOT / "traces"

DB_PATH = Path(os.getenv("CIA_DB_PATH", DATA_DIR / "campaigns.duckdb"))
LABELLED_PATH = DATA_DIR / "labelled_feedback.csv"
TRUTH_PATH = DATA_DIR / "feedback_truth.csv"
INDEX_PATH = DATA_DIR / "feedback_index.joblib"

# "auto" tries sentence-transformers and falls back to TF-IDF + SVD
EMBEDDER = os.getenv("CIA_EMBEDDER", "auto")
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ANTHROPIC_MODEL = os.getenv("CIA_ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL = os.getenv("CIA_OPENAI_MODEL", "gpt-4.1-mini")
