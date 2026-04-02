"""
app.py — Streamlit-based UI for the Google Ad Performance Analyzer.

Features:
  • Futuristic glassmorphism floating panel design
  • Chat-based interface with AI orb avatar
  • Semi-transparent metric cards with teal accents
  • Dynamic animated background with data network visualization
  • Floating glass input dock
  • Interactive charts with glass overlay styling
  • Conversation history for context-aware follow-ups
  • Cached data loading for speed
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
import re

from data_loader import load_dataset
from preprocess import preprocess
from llm_engine import query_llm, parse_chart_suggestion
from insight_generator import generate_rule_based_insights, generate_llm_insight
from guardrails import validate_query
from query_logger import log_query

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Ad Performance Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══ BACKGROUND ═══ */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0a0e1a;
        background-image:
            radial-gradient(ellipse 120% 80% at 20% 60%, rgba(0,180,216,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 100% 60% at 80% 30%, rgba(0,119,182,0.05) 0%, transparent 50%),
            radial-gradient(ellipse 80% 80% at 50% 90%, rgba(3,4,94,0.08) 0%, transparent 50%);
        min-height: 100vh;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }

    /* ═══ PARTICLES ═══ */
    [data-testid="stMain"]::before {
        content: '';
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 0;
        background-image:
            radial-gradient(2px 2px at 10% 20%, rgba(0,180,216,0.4), transparent),
            radial-gradient(2px 2px at 30% 70%, rgba(0,150,199,0.3), transparent),
            radial-gradient(1.5px 1.5px at 50% 40%, rgba(72,202,228,0.35), transparent),
            radial-gradient(2px 2px at 70% 80%, rgba(0,180,216,0.25), transparent),
            radial-gradient(1px 1px at 90% 30%, rgba(144,224,239,0.3), transparent),
            radial-gradient(2px 2px at 15% 90%, rgba(0,119,182,0.3), transparent),
            radial-gradient(1.5px 1.5px at 85% 60%, rgba(0,180,216,0.2), transparent),
            radial-gradient(1px 1px at 40% 15%, rgba(72,202,228,0.4), transparent),
            radial-gradient(2px 2px at 60% 50%, rgba(0,150,199,0.15), transparent),
            radial-gradient(1.5px 1.5px at 25% 45%, rgba(144,224,239,0.25), transparent);
        animation: particleDrift 30s ease-in-out infinite alternate;
    }
    @keyframes particleDrift {
        0% { transform: translate(0, 0) scale(1); opacity: 0.8; }
        50% { transform: translate(15px, -10px) scale(1.02); opacity: 1; }
        100% { transform: translate(5px, -5px) scale(1.01); opacity: 0.9; }
    }

    /* Grid overlay */
    [data-testid="stMain"]::after {
        content: '';
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 0;
        background-image:
            linear-gradient(rgba(0,180,216,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,180,216,0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: gridPulse 8s ease-in-out infinite;
    }
    @keyframes gridPulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.7; } }

    [data-testid="stMain"] > div { position: relative; z-index: 1; }

    /* ═══ TOOLBAR ═══ */
    .glass-toolbar {
        display: flex; align-items: center; justify-content: center; gap: 1rem;
        padding: 0.6rem 1.5rem; margin: 1rem auto 1.5rem; max-width: 420px;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 50px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .toolbar-btn {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.78rem; color: rgba(255,255,255,0.7);
        padding: 0.35rem 0.9rem; border-radius: 20px;
        transition: all 0.3s ease; cursor: pointer;
        background: transparent; border: none; font-family: 'Inter', sans-serif;
    }
    .toolbar-btn:hover { background: rgba(0,180,216,0.15); color: #48cae4; }
    .toolbar-btn.active { background: rgba(0,180,216,0.12); color: #90e0ef; }
    .toolbar-dots { font-size: 1rem; color: rgba(255,255,255,0.35); letter-spacing: 2px; }

    /* ═══ METRIC CHIP ═══ */
    .metric-chip {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(10px);
        border: 1px solid rgba(0,180,216,0.15); border-radius: 12px;
        padding: 0.65rem 0.8rem; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .metric-chip:hover {
        border-color: rgba(0,180,216,0.35);
        box-shadow: 0 4px 20px rgba(0,180,216,0.1);
        transform: translateY(-2px);
    }
    .metric-chip-label {
        font-size: 0.62rem; color: rgba(255,255,255,0.45);
        text-transform: uppercase; letter-spacing: 0.8px;
        margin-bottom: 0.15rem;
    }
    .metric-chip-value {
        font-size: 1.1rem; font-weight: 700; color: #00e676;
    }
    .metric-chip-value.teal { color: #48cae4; }
    .metric-chip-value.green { color: #00e676; }

    /* ═══ GLASS HUB ═══ */
    .glass-hub {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(30px) saturate(1.3);
        -webkit-backdrop-filter: blur(30px) saturate(1.3);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 24px;
        padding: 1.5rem 2rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 80px rgba(0,180,216,0.04),
            inset 0 1px 0 rgba(255,255,255,0.1), inset 0 -1px 0 rgba(255,255,255,0.02);
        position: relative; overflow: hidden;
        animation: floatHub 6s ease-in-out infinite;
    }
    @keyframes floatHub { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }

    .glass-hub::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), rgba(0,180,216,0.15), rgba(255,255,255,0.2), transparent);
    }
    .glass-hub::after {
        content: ''; position: absolute; top: 0; left: 0; bottom: 0; width: 1px;
        background: linear-gradient(180deg, rgba(255,255,255,0.15), transparent 70%);
    }

    .hub-header {
        font-size: 0.82rem; color: rgba(255,255,255,0.35);
        margin-bottom: 1.2rem; padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        font-weight: 400; letter-spacing: 0.3px;
    }

    /* ═══ CHAT BUBBLES ═══ */
    .user-msg { display: flex; justify-content: flex-end; gap: 0.6rem; align-items: flex-start; margin-bottom: 1rem; }
    .user-bubble {
        background: rgba(0,180,216,0.12); border: 1px solid rgba(0,180,216,0.2);
        border-radius: 18px 18px 4px 18px; padding: 0.85rem 1.2rem; max-width: 75%;
        color: rgba(255,255,255,0.9); font-size: 0.88rem; line-height: 1.55;
        box-shadow: 0 2px 12px rgba(0,180,216,0.08);
    }
    .user-avatar {
        width: 30px; height: 30px; border-radius: 50%;
        background: linear-gradient(135deg, rgba(0,180,216,0.3), rgba(0,119,182,0.4));
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; color: #48cae4; flex-shrink: 0;
        border: 1px solid rgba(0,180,216,0.25);
    }

    .ai-msg { display: flex; justify-content: flex-start; gap: 0.6rem; align-items: flex-start; margin-bottom: 1rem; }
    .ai-bubble {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px 18px 18px 4px; padding: 0.85rem 1.2rem; max-width: 80%;
        color: rgba(255,255,255,0.85); font-size: 0.88rem; line-height: 1.6;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }
    .ai-bubble h2, .ai-bubble h3, .ai-bubble h4 {
        color: #48cae4; font-size: 0.92rem; margin: 0.8rem 0 0.4rem 0; font-weight: 600;
    }
    .ai-bubble strong { color: #90e0ef; }
    .ai-bubble ul, .ai-bubble ol { padding-left: 1.2rem; margin: 0.5rem 0; }
    .ai-bubble li { margin: 0.25rem 0; }

    .ai-orb {
        width: 36px; height: 36px; border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #48cae4, #0077b6, #023e8a);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        box-shadow: 0 0 15px rgba(0,180,216,0.4), 0 0 30px rgba(0,180,216,0.15),
            inset 0 -2px 4px rgba(0,0,0,0.2), inset 0 2px 4px rgba(255,255,255,0.15);
        animation: orbGlow 3s ease-in-out infinite;
    }
    .ai-orb::after { content: '🤖'; font-size: 0.85rem; }
    @keyframes orbGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(0,180,216,0.4), 0 0 30px rgba(0,180,216,0.15); }
        50% { box-shadow: 0 0 25px rgba(0,180,216,0.6), 0 0 50px rgba(0,180,216,0.25); }
    }

    /* ═══ CHART AREA ═══ */
    .chart-glass {
        background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1rem; margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    /* ═══ FLOATING BADGES ═══ */
    .floating-badge {
        display: inline-flex; flex-direction: column; align-items: center;
        background: rgba(0,0,0,0.35); backdrop-filter: blur(15px);
        border: 1px solid rgba(0,180,216,0.2); border-radius: 14px;
        padding: 0.6rem 1.1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .badge-label {
        font-size: 0.6rem; color: rgba(255,255,255,0.45);
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    .badge-value { font-size: 1.2rem; font-weight: 700; color: #00e676; }

    /* ═══ INPUT DOCK ═══ */
    .dock-input .stTextInput > div > div > input {
        background: transparent !important; border: none !important;
        color: rgba(255,255,255,0.8) !important; font-size: 0.9rem !important;
        padding: 0.6rem 0.5rem !important; box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
    }
    .dock-input .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.3) !important; }
    .dock-input .stTextInput > div > div > input:focus { box-shadow: none !important; border: none !important; }
    .dock-input .stTextInput > div { border: none !important; box-shadow: none !important; }
    .dock-input .stTextInput label { display: none !important; }

    /* ═══ BUTTONS ═══ */
    .analyze-btn .stButton > button {
        background: linear-gradient(135deg, #0096c7, #0077b6) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.55rem 1.8rem !important; font-weight: 600 !important;
        font-size: 0.9rem !important; transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,150,199,0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .analyze-btn .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(0,180,216,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ═══ INSIGHTS ═══ */
    .insight-glass {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(0,180,216,0.1);
        border-left: 3px solid #0096c7; border-radius: 0 14px 14px 0;
        padding: 1rem 1.3rem; margin: 0.6rem 0;
        backdrop-filter: blur(10px); box-shadow: 0 2px 15px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    .insight-glass:hover { border-left-color: #48cae4; box-shadow: 0 4px 20px rgba(0,180,216,0.1); }
    .insight-glass h4 {
        color: #48cae4; margin: 0 0 0.4rem 0; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
    }
    .insight-glass p { color: rgba(255,255,255,0.75); font-size: 0.85rem; line-height: 1.5; margin: 0; }

    /* ═══ FLOATING ORB ═══ */
    .floating-orb-container { position: fixed; bottom: 2rem; right: 2rem; z-index: 1000; pointer-events: none; }
    .floating-orb-large {
        width: 70px; height: 70px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #48cae4, #0077b6 40%, #023e8a 70%, #03045e);
        box-shadow: 0 0 30px rgba(0,180,216,0.5), 0 0 60px rgba(0,180,216,0.2),
            0 0 100px rgba(0,180,216,0.08), inset 0 -3px 6px rgba(0,0,0,0.3),
            inset 0 3px 8px rgba(255,255,255,0.12);
        display: flex; align-items: center; justify-content: center;
        animation: orbFloat 4s ease-in-out infinite, orbGlowLarge 3s ease-in-out infinite;
        position: relative;
    }
    .floating-orb-large::before {
        content: ''; position: absolute; top: 8px; left: 12px;
        width: 18px; height: 12px;
        background: radial-gradient(ellipse, rgba(255,255,255,0.25), transparent);
        border-radius: 50%; transform: rotate(-25deg);
    }
    .floating-orb-large::after { content: '🤖'; font-size: 1.6rem; }
    @keyframes orbFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    @keyframes orbGlowLarge {
        0%, 100% { box-shadow: 0 0 30px rgba(0,180,216,0.5), 0 0 60px rgba(0,180,216,0.2); }
        50% { box-shadow: 0 0 45px rgba(0,180,216,0.7), 0 0 90px rgba(0,180,216,0.3); }
    }

    /* ═══ SIDEBAR ═══ */
    section[data-testid="stSidebar"] {
        background: rgba(10,14,26,0.95) !important;
        backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(0,180,216,0.08) !important; color: rgba(255,255,255,0.8) !important;
        border: 1px solid rgba(0,180,216,0.15) !important; border-radius: 10px !important;
        font-size: 0.82rem !important; padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important; font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,180,216,0.15) !important; border-color: rgba(0,180,216,0.3) !important;
    }

    /* ═══ WELCOME ═══ */
    .welcome-text { text-align: center; padding: 2rem 1rem; }
    .welcome-text h2 {
        font-size: 1.4rem; font-weight: 700;
        background: linear-gradient(135deg, #48cae4, #0096c7, #00b4d8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .welcome-text p {
        color: rgba(255,255,255,0.4); font-size: 0.85rem;
        max-width: 400px; margin: 0 auto; line-height: 1.6;
    }

    /* ═══ INSIGHTS SECTION ═══ */
    .insights-section-title {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.85rem; color: rgba(255,255,255,0.5);
        text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
        margin: 1.5rem 0 0.8rem;
    }
    .insights-section-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,180,216,0.2), transparent); }

    /* ═══ DIVIDER ═══ */
    .glass-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,180,216,0.2), rgba(0,119,182,0.15), transparent);
        border: none; margin: 1.5rem auto; max-width: 920px;
    }

    html { scroll-behavior: smooth; }

    @media (max-width: 768px) {
        .glass-hub { padding: 1rem; border-radius: 16px; }
        .metric-chip { padding: 0.5rem 0.6rem; }
        .glass-toolbar { max-width: 90%; }
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_data():
    """Load and preprocess data once, then cache."""
    raw = load_dataset()
    return preprocess(raw)


# ── Initialize session state ───────────────────────────────────────────────

def init_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


# ── Chart rendering ────────────────────────────────────────────────────────

def render_chart(chart_spec: dict, df: pd.DataFrame):
    """Render a chart with glassmorphism dark theme."""
    try:
        matplotlib.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#0d1117",
            "text.color": "white",
            "axes.labelcolor": "rgba(255,255,255,0.6)",
            "xtick.color": "#48cae4",
            "ytick.color": "#48cae4",
            "axes.edgecolor": "rgba(255,255,255,0.1)",
            "grid.color": "rgba(0,180,216,0.08)",
            "grid.alpha": 0.5,
        })

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        chart_type = chart_spec.get("chart_type", "bar")
        x = chart_spec.get("x", "")
        y = chart_spec.get("y", "")
        hue = chart_spec.get("hue")
        title = chart_spec.get("title", "Chart")

        if hue and hue.lower() in ("null", "none", ""):
            hue = None

        if x not in df.columns or y not in df.columns:
            st.warning(f"⚠️ Chart skipped — columns '{x}' or '{y}' not found.")
            return

        palette = ["#00b4d8", "#0096c7", "#0077b6", "#48cae4", "#90e0ef", "#00e676"]

        if chart_type == "bar":
            plot_df = df.groupby(x)[y].mean().reset_index().head(15)
            bars = sns.barplot(data=plot_df, x=x, y=y, palette=palette, ax=ax)
            for bar in bars.patches:
                bar.set_edgecolor('none')
                bar.set_alpha(0.85)
        elif chart_type == "line":
            plot_df = df.groupby(x)[y].mean().reset_index()
            sns.lineplot(data=plot_df, x=x, y=y, color="#00b4d8", linewidth=2.5, ax=ax)
            ax.fill_between(plot_df[x], plot_df[y], alpha=0.1, color="#00b4d8")
        elif chart_type == "pie":
            plot_df = df.groupby(x)[y].sum()
            ax.pie(plot_df.values, labels=plot_df.index, colors=palette,
                   autopct="%1.1f%%",
                   textprops={"color": "white", "fontsize": 9},
                   wedgeprops={"edgecolor": "rgba(255,255,255,0.1)", "linewidth": 0.5})
        else:
            plot_df = df.groupby(x)[y].mean().reset_index().head(15)
            sns.barplot(data=plot_df, x=x, y=y, palette=palette, ax=ax)

        ax.set_title(title, fontsize=12, fontweight="600", pad=12, color="#48cae4")
        ax.grid(True, alpha=0.15)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    except Exception as e:
        st.warning(f"⚠️ Could not render chart: {e}")


# ── Helper: render a single metric chip ────────────────────────────────────

def render_metric_chip(label: str, value: str, color_class: str = "green"):
    """Render a single glassmorphism metric chip."""
    st.markdown(
        f'<div class="metric-chip">'
        f'<div class="metric-chip-label">{label}</div>'
        f'<div class="metric-chip-value {color_class}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Main app ────────────────────────────────────────────────────────────────

def main():
    init_state()

    # Load data
    try:
        df = get_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ── Floating AI Orb (bottom-right) ──
    st.markdown(
        '<div class="floating-orb-container"><div class="floating-orb-large"></div></div>',
        unsafe_allow_html=True,
    )

    # ── Top Toolbar ──
    st.markdown(
        '<div class="glass-toolbar">'
        '<button class="toolbar-btn active">📊 View</button>'
        '<button class="toolbar-btn">🔍 Filters</button>'
        '<span class="toolbar-dots">•••</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI Metrics Row (using Streamlit columns) ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_metric_chip("CTR", f"{df['ctr'].mean()*100:.1f}%", "teal")
    with m2:
        render_metric_chip("Conversions", f"+{df['conversions'].sum():,.0f}", "green")
    with m3:
        render_metric_chip("Cost", f"${df['cost'].sum():,.0f}", "teal")
    with m4:
        render_metric_chip("ROI", f"{df['roi'].mean()*100:.0f}%", "green")
    with m5:
        render_metric_chip("Impressions", f"{df['impressions'].sum()/1e6:.1f}M", "teal")
    with m6:
        render_metric_chip("Revenue", f"${df['sale_amount'].sum():,.0f}", "green")

    st.markdown("")  # spacer

    # ── Main Glass Hub ──
    st.markdown('<div class="glass-hub">', unsafe_allow_html=True)
    st.markdown(
        '<div class="hub-header">✦ Your AI ad performance chatbot.</div>',
        unsafe_allow_html=True,
    )

    # ── Chat messages ──
    if st.session_state.chat_messages:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">'
                    f'<div class="user-bubble">{msg["content"]}</div>'
                    f'<div class="user-avatar">👤</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                content = msg["content"]
                # Convert markdown to HTML
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                content = re.sub(r'^## (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
                content = re.sub(r'^### (.*?)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
                content = re.sub(r'^- (.*?)$', r'<li>\1</li>', content, flags=re.MULTILINE)
                content = re.sub(
                    r'(<li>.*?</li>(\s*<li>.*?</li>)*)',
                    r'<ul>\1</ul>', content, flags=re.DOTALL,
                )
                content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')

                st.markdown(
                    f'<div class="ai-msg">'
                    f'<div class="ai-orb"></div>'
                    f'<div class="ai-bubble">{content}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Chart
                if msg.get("chart_spec"):
                    render_chart(msg["chart_spec"], df)

                # Insight
                if msg.get("insight"):
                    insight_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["insight"])
                    st.markdown(
                        f'<div class="insight-glass">'
                        f'<h4>💡 Proactive Insight</h4>'
                        f'<p>{insight_html}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        # Welcome state
        st.markdown(
            '<div class="welcome-text">'
            '<h2>Ask anything about your AI ad performance</h2>'
            '<p>Analyze campaigns, compare metrics, discover trends — '
            'powered by advanced AI that understands your Google Ads data.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Floating badges showing key metrics
        roi_val = f"{df['roi'].mean()*100:.0f}%"
        imp_val = f"{df['impressions'].sum()/1e6:.1f}M"
        conv_val = f"{df['conversions'].sum():,.0f}"

        bcol1, bcol2, bcol3 = st.columns([1, 3, 1])
        with bcol2:
            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(
                    f'<div class="floating-badge">'
                    f'<span class="badge-label">ROI</span>'
                    f'<span class="badge-value">{roi_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f'<div class="floating-badge">'
                    f'<span class="badge-label">Impressions</span>'
                    f'<span class="badge-value">{imp_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    f'<div class="floating-badge">'
                    f'<span class="badge-label">Conversions</span>'
                    f'<span class="badge-value">{conv_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)  # close glass-hub

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🎯 Quick Queries")
        suggestions = [
            "Which campaigns have the highest CTR?",
            "What is the trend in spend over time?",
            "Which device performs best by conversions?",
            "Top 5 keywords by ROI",
            "Compare weekend vs weekday performance",
            "Which ads have high spend but low conversions?",
            "What is the average cost per conversion?",
            "Show monthly conversion trends",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                st.session_state["prefill_question"] = s

        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- **Rows:** {len(df):,}")
        st.markdown(f"- **Columns:** {len(df.columns)}")
        st.markdown(
            f"- **Date range:** {df['ad_date'].min().strftime('%Y-%m-%d')} → "
            f"{df['ad_date'].max().strftime('%Y-%m-%d')}"
        )
        st.markdown(f"- **Devices:** {', '.join(df['device'].unique())}")
        st.markdown(f"- **Keywords:** {df['keyword'].nunique()}")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_raw = st.checkbox("Show raw data explorer", value=False)
        show_proactive = st.checkbox("Show proactive insights", value=True)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            st.rerun()

    # Raw data explorer
    if show_raw:
        with st.expander("📁 Raw Data Explorer", expanded=False):
            st.dataframe(df, use_container_width=True, height=300)

    # ── Question Input ──
    prefill = st.session_state.pop("prefill_question", "")

    dock_cols = st.columns([6, 1])
    with dock_cols[0]:
        st.markdown('<div class="dock-input">', unsafe_allow_html=True)
        question = st.text_input(
            "Type a message...",
            value=prefill,
            placeholder="Have any questions about your AI ad performance?",
            label_visibility="collapsed",
            key="question_input",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with dock_cols[1]:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        ask_clicked = st.button("🚀 Analyze", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if ask_clicked and question:
        is_valid, validation_msg = validate_query(question)
        if not is_valid:
            st.warning(validation_msg)
            return

        st.session_state.chat_messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Analyzing your data with AI..."):
            try:
                answer = query_llm(
                    question=question,
                    df=df,
                    conversation_history=st.session_state.conversation_history,
                )

                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": answer}
                )

                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = (
                        st.session_state.conversation_history[-10:]
                    )

                chart_spec = parse_chart_suggestion(answer)

                insight_text = ""
                if show_proactive:
                    try:
                        insight_text = generate_llm_insight(df, question, answer)
                    except Exception:
                        rule_insights = generate_rule_based_insights(df)
                        insight_text = rule_insights[0] if rule_insights else ""

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chart_spec": chart_spec,
                    "insight": insight_text,
                })

                log_query(question, answer, insight_text)

            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                log_query(question, "", error=error_msg)

        st.rerun()

    # ── Proactive insights panel ──
    if show_proactive:
        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="insights-section-title">'
            '<span>🔮 Proactive Insights</span>'
            '<div class="insights-section-line"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        insights = generate_rule_based_insights(df)
        if insights:
            for insight in insights[:4]:
                insight_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', insight)
                st.markdown(
                    f'<div class="insight-glass">'
                    f'<p>{insight_html}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No automatic insights detected at this time.")


if __name__ == "__main__":
    main()
