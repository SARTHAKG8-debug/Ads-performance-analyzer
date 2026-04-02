"""
insight_generator.py — Automatically generate proactive insights from the data.

After every user query, this module analyses the full dataset and produces
at least one genuinely useful, actionable insight the user did NOT ask for.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from openai import OpenAI


# ── Rule-based heuristic insights ──────────────────────────────────────────

def _high_spend_low_conversion(df: pd.DataFrame) -> list[str]:
    """Flag ads with above-average cost but below-average conversions."""
    insights = []
    if {"cost", "conversions", "ad_id"}.issubset(df.columns):
        avg_cost = df["cost"].mean()
        avg_conv = df["conversions"].mean()
        flagged = df[(df["cost"] > avg_cost * 1.2) & (df["conversions"] < avg_conv * 0.8)]
        if len(flagged) > 0:
            count = len(flagged)
            avg_waste = (flagged["cost"] - (flagged["conversions"] * (avg_cost / avg_conv))).mean()
            insights.append(
                f"⚠️ **{count} ads** have high spend (>20% above average) but low conversions "
                f"(<20% below average). These may need budget reallocation or creative optimization."
            )
    return insights


def _declining_ctr_trend(df: pd.DataFrame) -> list[str]:
    """Detect if CTR has been declining over the last 7 days of data."""
    insights = []
    if {"ad_date", "ctr"}.issubset(df.columns):
        df_sorted = df.dropna(subset=["ad_date", "ctr"]).sort_values("ad_date")
        if len(df_sorted) > 0:
            last_date = df_sorted["ad_date"].max()
            seven_days_ago = last_date - pd.Timedelta(days=7)
            recent = df_sorted[df_sorted["ad_date"] >= seven_days_ago]
            earlier = df_sorted[df_sorted["ad_date"] < seven_days_ago]
            if len(recent) > 5 and len(earlier) > 5:
                recent_ctr = recent["ctr"].mean()
                earlier_ctr = earlier["ctr"].mean()
                if recent_ctr < earlier_ctr * 0.9:
                    drop_pct = ((earlier_ctr - recent_ctr) / earlier_ctr) * 100
                    insights.append(
                        f"📉 **CTR has declined ~{drop_pct:.1f}%** in the last 7 days of data "
                        f"(from {earlier_ctr:.4f} to {recent_ctr:.4f}). Consider refreshing ad creatives."
                    )
    return insights


def _best_performing_device(df: pd.DataFrame) -> list[str]:
    """Identify the best-performing device by conversion rate."""
    insights = []
    if {"device", "conversions", "clicks"}.issubset(df.columns):
        device_stats = (
            df.groupby("device")
            .agg(total_clicks=("clicks", "sum"), total_conversions=("conversions", "sum"))
            .assign(conv_rate=lambda x: x["total_conversions"] / x["total_clicks"])
            .sort_values("conv_rate", ascending=False)
        )
        if len(device_stats) >= 2:
            best = device_stats.index[0]
            worst = device_stats.index[-1]
            best_rate = device_stats.loc[best, "conv_rate"] * 100
            worst_rate = device_stats.loc[worst, "conv_rate"] * 100
            insights.append(
                f"📱 **'{best}'** has the highest conversion rate ({best_rate:.2f}%), "
                f"while **'{worst}'** lags behind ({worst_rate:.2f}%). "
                f"Consider shifting budget toward {best}."
            )
    return insights


def _top_keyword_roi(df: pd.DataFrame) -> list[str]:
    """Find the keyword with the best ROI."""
    insights = []
    if {"keyword", "roi"}.issubset(df.columns):
        kw_roi = (
            df.dropna(subset=["roi"])
            .groupby("keyword")["roi"]
            .mean()
            .sort_values(ascending=False)
        )
        if len(kw_roi) >= 2:
            best_kw = kw_roi.index[0]
            best_val = kw_roi.iloc[0] * 100
            worst_kw = kw_roi.index[-1]
            worst_val = kw_roi.iloc[-1] * 100
            insights.append(
                f"🔑 Keyword **'{best_kw}'** delivers the highest avg ROI ({best_val:.1f}%), "
                f"while **'{worst_kw}'** has the lowest ({worst_val:.1f}%). "
                f"Review keyword bidding strategy."
            )
    return insights


def _weekend_vs_weekday(df: pd.DataFrame) -> list[str]:
    """Compare weekend vs weekday performance."""
    insights = []
    if {"ad_date", "conversions", "cost"}.issubset(df.columns):
        df_temp = df.dropna(subset=["ad_date"]).copy()
        df_temp["is_weekend"] = df_temp["ad_date"].dt.dayofweek >= 5
        weekend = df_temp[df_temp["is_weekend"]]
        weekday = df_temp[~df_temp["is_weekend"]]
        if len(weekend) > 10 and len(weekday) > 10:
            we_cpc = weekend["cost"].sum() / weekend["conversions"].sum() if weekend["conversions"].sum() > 0 else 0
            wd_cpc = weekday["cost"].sum() / weekday["conversions"].sum() if weekday["conversions"].sum() > 0 else 0
            if we_cpc > 0 and wd_cpc > 0:
                if we_cpc > wd_cpc * 1.15:
                    insights.append(
                        f"📅 Weekend cost-per-conversion (${we_cpc:.2f}) is **{((we_cpc/wd_cpc)-1)*100:.0f}% higher** "
                        f"than weekdays (${wd_cpc:.2f}). Consider reducing weekend ad spend."
                    )
                elif wd_cpc > we_cpc * 1.15:
                    insights.append(
                        f"📅 Weekday cost-per-conversion (${wd_cpc:.2f}) is **{((wd_cpc/we_cpc)-1)*100:.0f}% higher** "
                        f"than weekends (${we_cpc:.2f}). Weekends might be more efficient."
                    )
    return insights


# ── Aggregate all heuristic insights ────────────────────────────────────────

def generate_rule_based_insights(df: pd.DataFrame) -> list[str]:
    """
    Run all heuristic checks and return a list of insight strings.
    """
    insights: list[str] = []
    insights.extend(_high_spend_low_conversion(df))
    insights.extend(_declining_ctr_trend(df))
    insights.extend(_best_performing_device(df))
    insights.extend(_top_keyword_roi(df))
    insights.extend(_weekend_vs_weekday(df))
    return insights


# ── LLM-powered contextual insight ─────────────────────────────────────────

def generate_llm_insight(
    df: pd.DataFrame,
    user_question: str,
    llm_answer: str,
) -> str:
    """
    Ask the LLM to generate one additional, genuinely useful insight
    the user did NOT ask for, based on the data and the current Q&A context.
    """
    # Build a compact data summary
    summary_lines = []
    if "device" in df.columns:
        summary_lines.append(f"Devices: {df['device'].value_counts().to_dict()}")
    if "keyword" in df.columns:
        summary_lines.append(f"Keywords: {df['keyword'].value_counts().head(5).to_dict()}")
    if {"cost", "conversions"}.issubset(df.columns):
        summary_lines.append(f"Avg cost: ${df['cost'].mean():.2f}, Avg conversions: {df['conversions'].mean():.1f}")
    if "roi" in df.columns:
        summary_lines.append(f"Avg ROI: {df['roi'].mean()*100:.1f}%")

    data_context = "\n".join(summary_lines)

    prompt = f"""
You are a senior digital marketing analyst.  
A user asked: "{user_question}"
The system answered: "{llm_answer[:500]}"

Here is a summary of the overall dataset:
{data_context}

Based on the FULL dataset context, generate exactly ONE proactive, actionable 
insight that is DIFFERENT from what the user asked.  The insight should be 
genuinely useful for an advertiser optimizing their Google Ads campaigns.

Format: Start with an emoji, then bold the key metric, then explain the 
implication and recommended action in 1-2 sentences.
"""

    try:
        if LLM_PROVIDER == "gemini":
            client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            model = GEMINI_MODEL
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            model = OPENAI_MODEL

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to rule-based if LLM fails
        rule_insights = generate_rule_based_insights(df)
        return rule_insights[0] if rule_insights else "💡 Tip: Review your highest-cost ads for conversion efficiency."
