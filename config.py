"""
config.py — Centralized configuration for the Google Ad Performance Analyzer.

Loads settings from .env file (local) or Streamlit secrets (cloud deployment).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Get a config value from Streamlit secrets (cloud) or env vars (local)."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # Fall back to environment variables
    return os.getenv(key, default)


# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / _get_secret("DATASET_PATH", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")

# ── LLM provider selection ──────────────────────────────────────────────────
LLM_PROVIDER = _get_secret("LLM_PROVIDER", "openai").strip().lower()

# ── OpenAI settings ─────────────────────────────────────────────────────────
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")

# ── Gemini settings ─────────────────────────────────────────────────────────
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-2.0-flash")

# ── App behaviour ───────────────────────────────────────────────────────────
MAX_SAMPLE_ROWS = 5        # rows sent to the LLM as context
CACHE_TTL_SECONDS = 300    # Streamlit cache lifetime
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
