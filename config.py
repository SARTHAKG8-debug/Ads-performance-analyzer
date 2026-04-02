"""
config.py — Centralized configuration for the Google Ad Performance Analyzer.

Loads settings from .env file and provides validated defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_PATH", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")

# ── LLM provider selection ──────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

# ── OpenAI settings ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Gemini settings ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── App behaviour ───────────────────────────────────────────────────────────
MAX_SAMPLE_ROWS = 5        # rows sent to the LLM as context
CACHE_TTL_SECONDS = 300    # Streamlit cache lifetime
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
