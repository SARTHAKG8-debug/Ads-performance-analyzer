"""
query_logger.py — Log every user query, LLM response, and timestamp.

Logs are stored as JSONL files in the logs/ directory for audit & analysis.
"""

import json
import datetime
from pathlib import Path
from config import LOG_DIR


LOG_FILE = LOG_DIR / "query_log.jsonl"


def log_query(
    question: str,
    answer: str,
    insight: str = "",
    error: str = "",
) -> None:
    """
    Append a structured log entry for each user interaction.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    answer : str
        The LLM's response.
    insight : str
        The proactive insight generated.
    error : str
        Error message, if any.
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer_length": len(answer),
        "insight": insight[:200] if insight else "",
        "error": error,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
