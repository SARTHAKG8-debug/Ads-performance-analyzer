"""
llm_engine.py — LLM-powered natural language query engine for Google Ads data.

Responsibilities:
  • Build a structured prompt containing the dataset schema, sample rows,
    and strict grounding instructions.
  • Send queries to the configured LLM (OpenAI GPT or Google Gemini).
  • Parse the response into a structured format (summary, findings, table).
  • Provide context-aware follow-up support via conversation history.
"""

import json
import textwrap
from io import StringIO
from typing import Optional

import pandas as pd
from openai import OpenAI

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_SAMPLE_ROWS,
)


# ── Prompt templates ────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert digital marketing data analyst.  You have access to a
    Google Ads campaign dataset whose schema and sample rows are provided below.

    **STRICT RULES — follow them without exception:**
    1. ONLY use information present in the dataset.  Never invent data.
    2. If the answer cannot be determined from the data, say so explicitly.
    3. All statistics you quote must be directly calculable from the columns.
    4. Do NOT hallucinate campaign names, dates, or numbers.

    **Response format (use exactly these markdown headers):**

    ## Summary
    (1–2 sentence answer)

    ## Key Findings
    - bullet 1
    - bullet 2
    - …

    ## Data Table (optional — include only when the user asks for a list,
       ranking, or comparison)
    | Column A | Column B | … |
    |----------|----------|---|
    | val      | val      | … |

    ## Visualization Suggestion (optional)
    If a chart would help, describe what to plot (chart type, x-axis, y-axis,
    grouping). Use this exact JSON format on a single line:
    ```json
    {{"chart_type": "bar|line|pie", "x": "column", "y": "column", "hue": "column_or_null", "title": "Chart title"}}
    ```

    ---
    **Dataset Schema:**
    {schema}

    **Sample Rows (first {n_rows}):**
    {sample_rows}

    **Column Statistics:**
    {col_stats}
""")


# ── Helper: Build context ──────────────────────────────────────────────────

def _build_schema_text(df: pd.DataFrame) -> str:
    """Create a readable schema description."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        null_pct = df[col].isna().mean() * 100
        lines.append(f"  • {col}  ({dtype}, {n_unique} unique, {null_pct:.1f}% null)")
    return "\n".join(lines)


def _build_col_stats(df: pd.DataFrame) -> str:
    """Descriptive statistics for numeric columns."""
    desc = df.describe(include="all").round(2)
    return desc.to_string()


def _build_system_message(df: pd.DataFrame) -> str:
    """Assemble the full system prompt with live data context."""
    schema_text = _build_schema_text(df)
    sample_text = df.head(MAX_SAMPLE_ROWS).to_string(index=False)
    stats_text = _build_col_stats(df)

    return SYSTEM_PROMPT.format(
        schema=schema_text,
        n_rows=MAX_SAMPLE_ROWS,
        sample_rows=sample_text,
        col_stats=stats_text,
    )


# ── LLM Clients ────────────────────────────────────────────────────────────

def _get_openai_client() -> OpenAI:
    """Instantiate the OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_gemini_client() -> OpenAI:
    """
    Use the OpenAI-compatible API endpoint for Google Gemini.
    This lets us reuse the same client interface.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# ── Core query function ────────────────────────────────────────────────────

def query_llm(
    question: str,
    df: pd.DataFrame,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Send a natural-language question to the LLM and return its text answer.

    Parameters
    ----------
    question : str
        User's plain-English question about the ad data.
    df : pd.DataFrame
        The preprocessed DataFrame (used for schema / sample context).
    conversation_history : list[dict] or None
        Previous messages for context-aware follow-ups.  Each dict has
        keys ``role`` ('user' | 'assistant') and ``content``.

    Returns
    -------
    str
        The LLM's markdown-formatted answer.
    """
    system_msg = _build_system_message(df)

    messages = [{"role": "system", "content": system_msg}]

    # Append conversation history for follow-up awareness
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": question})

    # Select provider
    if LLM_PROVIDER == "gemini":
        client = _get_gemini_client()
        model = GEMINI_MODEL
    else:
        client = _get_openai_client()
        model = OPENAI_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,       # low temp → factual, grounded answers
        max_tokens=2000,
    )

    return response.choices[0].message.content


# ── Response parser ─────────────────────────────────────────────────────────

def parse_chart_suggestion(response: str) -> Optional[dict]:
    """
    Extract the JSON chart suggestion from the LLM response, if present.

    Returns None if no suggestion is found.
    """
    import re

    pattern = r'```json\s*(\{.*?"chart_type".*?\})\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None
