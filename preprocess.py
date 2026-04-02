"""
preprocess.py — Clean and prepare the raw Google Ads data for analysis.

Pipeline:
  1. Normalize column names (lowercase, underscores)
  2. Parse & unify date formats
  3. Strip currency symbols and cast numeric columns
  4. Handle missing values sensibly
  5. Feature-engineer CTR, CPC, cost-per-conversion, ROI
  6. Normalize categorical text (location, device, campaign name)
"""

import re
import pandas as pd
import numpy as np


# ── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace spaces / special chars with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse dates that come in mixed formats:
      • 2024-11-16   (ISO)
      • 20-11-2024   (DD-MM-YYYY)
      • 2024/11/16   (YYYY/MM/DD)
    """
    return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")


def _clean_currency(series: pd.Series) -> pd.Series:
    """Remove $ signs, commas, whitespace, then convert to float."""
    return (
        series
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )


def _normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase and strip whitespace from text columns."""
    return series.astype(str).str.strip().str.lower()


def _standardize_campaign_name(series: pd.Series) -> pd.Series:
    """
    Map all spelling variations of the campaign name to a
    single canonical form: 'data analytics course'.
    """
    return (
        series
        .str.lower()
        .str.strip()
        .replace(
            {
                "dataanalyticscourse": "data analytics course",
                "data anlytics corse": "data analytics course",
                "data analytcis course": "data analytics course",
                "data analytics corse": "data analytics course",
            }
        )
    )


def _standardize_location(series: pd.Series) -> pd.Series:
    """Map all location variants to the canonical spelling."""
    mapping = {
        "hyderabad": "hyderabad",
        "hyderbad": "hyderabad",
        "hydrebad": "hyderabad",
    }
    return series.str.lower().str.strip().map(mapping).fillna(series.str.lower().str.strip())


# ── Main preprocessing function ────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline.  Returns a new DataFrame — the
    original is never mutated.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from data_loader.load_dataset().

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered DataFrame ready for analysis.
    """
    df = df.copy()

    # 1. Column names
    df = _normalize_columns(df)

    # 2. Date parsing
    if "ad_date" in df.columns:
        df["ad_date"] = _parse_dates(df["ad_date"])

    # 3. Currency / numeric columns
    if "cost" in df.columns:
        df["cost"] = _clean_currency(df["cost"])

    if "sale_amount" in df.columns:
        df["sale_amount"] = _clean_currency(df["sale_amount"])

    numeric_cols = ["clicks", "impressions", "leads", "conversions", "conversion_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Text normalization
    if "campaign_name" in df.columns:
        df["campaign_name"] = _standardize_campaign_name(df["campaign_name"])

    if "location" in df.columns:
        df["location"] = _standardize_location(df["location"])

    if "device" in df.columns:
        df["device"] = _normalize_text(df["device"])

    if "keyword" in df.columns:
        df["keyword"] = _normalize_text(df["keyword"])

    # 5. Handle missing values
    #    – Numeric: fill with column median (robust to outliers)
    #    – Categorical: leave as-is (NaN → "unknown" only if needed downstream)
    for col in ["clicks", "impressions", "cost", "leads", "conversions", "sale_amount"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 6. Feature engineering
    if {"clicks", "impressions"}.issubset(df.columns):
        df["ctr"] = (df["clicks"] / df["impressions"]).round(4)
        df["ctr"] = df["ctr"].replace([np.inf, -np.inf], np.nan)

    if {"cost", "clicks"}.issubset(df.columns):
        df["cpc"] = (df["cost"] / df["clicks"]).round(2)
        df["cpc"] = df["cpc"].replace([np.inf, -np.inf], np.nan)

    if {"cost", "conversions"}.issubset(df.columns):
        df["cost_per_conversion"] = (df["cost"] / df["conversions"]).round(2)
        df["cost_per_conversion"] = df["cost_per_conversion"].replace([np.inf, -np.inf], np.nan)

    if {"sale_amount", "cost"}.issubset(df.columns):
        df["roi"] = ((df["sale_amount"] - df["cost"]) / df["cost"]).round(4)
        df["roi"] = df["roi"].replace([np.inf, -np.inf], np.nan)

    # 7. Sort by date
    if "ad_date" in df.columns:
        df = df.sort_values("ad_date").reset_index(drop=True)

    # Drop the old conversion_rate column if it exists (we recalculated ctr)
    if "conversion_rate" in df.columns:
        df = df.drop(columns=["conversion_rate"])

    print(f"✅  Preprocessing complete — {len(df):,} rows, {len(df.columns)} columns")
    return df
