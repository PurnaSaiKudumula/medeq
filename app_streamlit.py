"""
app_streamlit.py — MedEQ Medical Equipment Predictive Maintenance Dashboard
=============================================================================

Medical + Clinical + AI + Enterprise Dashboard style.

Four views:
  1. Device Risk List — Fleet overview, ranked alerts, SHAP, SOP generation
  2. What-If Simulator — Interactive sliders for live risk prediction
  3. Model Comparison — Realistic vs Optimized model transparency
  4. RUL Estimator — Interactive remaining-useful-life countdown

RUN:  streamlit run app_streamlit.py
Backend: uvicorn src.api:app --reload (port 8000)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import re
import time
import html

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

API_BASE_URL = "http://127.0.0.1:8000"

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="MedEQ — Medical Equipment Predictive Maintenance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# COMPLETE DESIGN SYSTEM — CSS
# ===========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --navy: #0B2438;
        --navy-hover: #123A57;
        --teal: #2DD4BF;
        --teal-light: rgba(45, 212, 191, 0.16);
        --blue: #60A5FA;
        --blue-light: rgba(96, 165, 250, 0.16);
        --low: #34D399;
        --low-light: rgba(52, 211, 153, 0.16);
        --medium: #FBBF24;
        --medium-light: rgba(251, 191, 36, 0.16);
        --high: #FB7185;
        --high-light: rgba(251, 113, 133, 0.16);
        --canvas: #081A24;
        --surface: #0E2A36;
        --surface-2: #143643;
        --border: #1E4857;
        --ink: #E4F1F3;
        --muted: #8FB6C0;
        --muted-light: #5E7F8A;
        --track: rgba(148, 183, 194, 0.22);
    }

    * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(1100px 620px at 12% -12%, rgba(37, 99, 235, 0.16) 0%, rgba(37, 99, 235, 0) 55%),
            radial-gradient(1000px 620px at 100% 0%, rgba(15, 118, 110, 0.22) 0%, rgba(15, 118, 110, 0) 55%),
            linear-gradient(160deg, #0B2B3A 0%, #081A24 48%, #0A3A34 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stMainMenu"] { visibility: hidden; }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] { background: transparent; }
    [data-testid="stVerticalBlock"] { background: transparent; }

    .block-container {
        max-width: 1200px;
        padding: 2rem 2.5rem 4rem;
        background: transparent;
    }

    h1, h2, h3, h4, h5, h6 { color: var(--ink) !important; letter-spacing: -0.01em; }
    p, li, span { color: inherit; }

    /* ====== APP HEADER ====== */
    .app-header {
        background: linear-gradient(135deg, #0D3048 0%, #0B6B5F 100%);
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    }
    .app-header h1 {
        color: #F3FFFD !important;
        font-size: clamp(1.9rem, 3vw, 2.8rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }
    .app-header .subtitle {
        color: #9FD8D0;
        font-size: 1.05rem;
        margin: 0.4rem 0 0;
        font-weight: 400;
    }
    .app-header .supporting {
        color: #7FB6AF;
        font-size: 0.88rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* ====== AI PIPELINE STRIP ====== */
    .pipeline-strip {
        display: flex;
        align-items: center;
        gap: 0;
        margin-bottom: 1.25rem;
        overflow-x: auto;
        padding: 0.5rem 0;
    }
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.85rem;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--ink);
    }
    .pipeline-step.navy { background: #12354E; color: #DFF6F2; border-color: #1E4857; }
    .pipeline-step.teal { background: rgba(45, 212, 191, 0.16); color: var(--teal); border-color: rgba(45, 212, 191, 0.4); }
    .pipeline-step.blue { background: rgba(96, 165, 250, 0.16); color: var(--blue); border-color: rgba(96, 165, 250, 0.4); }
    .pipeline-arrow {
        color: var(--muted-light);
        font-size: 0.9rem;
        margin: 0 0.25rem;
        flex-shrink: 0;
    }

    /* ====== TABS ====== */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 2px solid var(--border);
    }
    [data-testid="stTabs"] button {
        color: var(--muted);
        font-weight: 700;
        padding: 0.75rem 1rem;
    }
    [data-testid="stTabs"] button p,
    [data-testid="stTabs"] button span,
    [data-baseweb="tab"] p,
    [data-baseweb="tab"] span { color: var(--muted) !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
    [data-testid="stTabs"] button[aria-selected="true"] p,
    [data-testid="stTabs"] button[aria-selected="true"] span,
    [data-baseweb="tab"][aria-selected="true"] p,
    [data-baseweb="tab"][aria-selected="true"] span { color: var(--teal) !important; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--teal); }

    /* ====== FLEET OVERVIEW CARDS ====== */
    .fleet-overview { margin-bottom: 1.5rem; }
    .fleet-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.6rem;
    }
    .fleet-cards { display: flex; gap: 14px; flex-wrap: wrap; }
    .fleet-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        flex: 1;
        min-width: 140px;
    }
    .fleet-card .fleet-value {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }
    .fleet-card .fleet-desc {
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 4px;
        font-weight: 500;
    }
    .fleet-card.total .fleet-value { color: var(--teal); }
    .fleet-card.high .fleet-value { color: var(--high); }
    .fleet-card.med .fleet-value { color: var(--medium); }
    .fleet-card.low .fleet-value { color: var(--low); }

    /* ====== SECTION HEADERS ====== */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--teal);
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    h5 {
        color: var(--teal) !important;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        letter-spacing: -0.01em;
    }

    /* ====== SYSTEM STATUS PILL ====== */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        border: 1px solid var(--border);
        background: var(--surface);
    }
    .status-pill .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-pill.online .dot { background: var(--low); }
    .status-pill.online { color: var(--low); }
    .status-pill.offline .dot { background: var(--high); }
    .status-pill.offline { color: var(--high); }
    .status-pill.warning .dot { background: var(--medium); }
    .status-pill.warning { color: var(--medium); }
    .status-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
        margin: 0 0 0.25rem 0;
    }

    /* ====== ALERT CARDS ====== */
    .alert-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15,39,71,0.04);
        border-left: 5px solid var(--border);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .alert-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15,39,71,0.08);
    }
    .alert-card.critical { border-left-color: var(--high); }
    .alert-card.warning  { border-left-color: var(--medium); }
    .alert-card.routine  { border-left-color: var(--teal); }

    .alert-device-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 3px;
    }
    .alert-classification {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 8px;
    }

    /* ====== WARD BADGES ====== */
    .ward-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .ward-badge.icu { background: var(--blue-light); color: #9CCBFF; }
    .ward-badge.general { background: var(--teal-light); color: var(--teal); }
    .ward-badge.storage { background: rgba(148, 163, 184, 0.18); color: #A9C4CE; }

    /* ====== RISK BADGES ====== */
    .risk-badge {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        padding: 6px 16px;
        border-radius: 10px;
        font-weight: 700;
        min-width: 80px;
    }
    .risk-badge .risk-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .risk-badge .risk-pct {
        font-size: 1.3rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .risk-badge.critical { background: var(--high-light); color: #FCA5B5; }
    .risk-badge.warning  { background: var(--medium-light); color: #FCDB7A; }
    .risk-badge.routine  { background: var(--low-light); color: #7FE0B8; }

    /* ====== PRIORITY SCORE ====== */
    .priority-score {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--teal);
        line-height: 1;
    }
    .priority-label {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ====== METRIC DISPLAY ====== */
    .metric-block .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--teal);
        line-height: 1.1;
    }
    .metric-block .metric-label {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 500;
    }

    /* ====== RUL (REMAINING USEFUL LIFE) BAR ====== */
    .rul-bar-wrap {
        margin-top: 8px;
        max-width: 240px;
    }
    .rul-bar-track {
        height: 8px;
        border-radius: 4px;
        background: var(--track);
        overflow: hidden;
    }
    .rul-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .rul-bar-caption {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 4px;
        font-weight: 600;
    }

    /* ====== DRIVER ROWS ====== */
    .driver-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid var(--border);
    }
    .driver-row:last-child { border-bottom: none; }
    .driver-feature { font-weight: 600; color: var(--ink); font-size: 0.9rem; }
    .driver-impact { font-weight: 700; font-size: 0.88rem; }
    .driver-impact.positive { color: var(--high); }
    .driver-impact.negative { color: var(--low); }

    .driver-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 3px;
    }
    .driver-bar.positive { background: var(--high); }
    .driver-bar.negative { background: var(--low); }

    /* ====== ACTION BADGES ====== */
    .action-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .action-badge.critical { background: var(--high-light); color: #FCA5B5; border: 1px solid rgba(251,113,133,0.3); }
    .action-badge.warning  { background: var(--medium-light); color: #FCDB7A; border: 1px solid rgba(251,191,36,0.3); }
    .action-badge.routine  { background: var(--low-light); color: #7FE0B8; border: 1px solid rgba(52,211,153,0.3); }

    /* ====== WHAT-IF RESULT CARD ====== */
    .whatif-result {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px;
        box-shadow: 0 2px 8px rgba(15,39,71,0.05);
    }
    .whatif-risk-large {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .whatif-risk-label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    /* ====== SOP WORK ORDER ====== */
    .sop-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    .sop-header {
        background: linear-gradient(135deg, #0D3048 0%, #0B6B5F 100%);
        color: #F3FFFD;
        padding: 16px 24px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .sop-body { padding: 24px; }
    .sop-section-title {
        color: var(--teal);
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 16px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 2px solid var(--teal-light);
    }
    .sop-section-title:first-child { margin-top: 0; }
    .sop-text { color: var(--ink); font-size: 0.92rem; line-height: 1.6; }
    .sop-info { color: var(--blue); font-weight: 500; }
    .sop-urgency-high { color: var(--high); font-weight: 700; font-size: 1.05rem; }
    .sop-urgency-med  { color: var(--medium); font-weight: 700; font-size: 1.05rem; }
    .sop-urgency-low  { color: var(--low); font-weight: 700; font-size: 1.05rem; }

    /* ====== MODEL COMPARISON CARDS ====== */
    .model-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        border-top: 4px solid var(--border);
    }
    .model-card.realistic { border-top-color: var(--teal); }
    .model-card.optimized { border-top-color: var(--blue); }
    .model-title {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 16px;
    }
    .model-card.realistic .model-title { color: var(--teal); }
    .model-card.optimized .model-title { color: var(--blue); }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .metric-tile {
        background: rgba(12, 40, 52, 0.6);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-tile .mt-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--ink);
    }
    .metric-tile .mt-label {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* ====== BUTTONS ====== */
    .stButton > button, .stDownloadButton > button {
        border: 1px solid transparent;
        border-radius: 8px;
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        min-height: 2.65rem;
        transition: background 0.15s, border-color 0.15s, color 0.15s, filter 0.15s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.1);
        border-color: rgba(45, 212, 191, 0.6);
        color: #FFFFFF;
    }
    .stButton > button:disabled, .stDownloadButton > button:disabled {
        background: var(--surface-2);
        color: var(--muted);
        border-color: var(--border);
    }
    .stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {
        background: var(--surface);
        color: var(--teal);
        border: 1px solid rgba(45, 212, 191, 0.5);
    }
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {
        background: var(--teal-light);
        border-color: var(--teal);
        color: var(--teal);
    }
    .stButton > button p, .stButton > button span,
    .stDownloadButton > button p, .stDownloadButton > button span {
        color: inherit !important;
    }

    /* ====== RAW HTML BUTTON (Calculate Risk inside markdown) ====== */
    .action-calc {
        display: inline-block;
        width: 100%;
        text-align: center;
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        color: #FFFFFF !important;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
        border: 1px solid transparent;
        transition: filter 0.15s;
    }
    .action-calc:hover { filter: brightness(1.1); }

    /* ====== SLIDERS ====== */
    [data-testid="stSlider"] label p,
    [data-testid="stSelectSlider"] label p,
    [data-testid="stSelectbox"] label p {
        color: var(--ink) !important;
        font-weight: 600;
    }
    [data-baseweb="slider"] > div { color: var(--teal) !important; }
    [data-baseweb="select"] > div {
        background: var(--surface-2) !important;
        border-color: var(--border) !important;
        color: var(--ink) !important;
    }
    [data-baseweb="select"] * { color: var(--ink) !important; }
    [data-baseweb="popover"] [role="listbox"] { background: var(--surface-2) !important; }
    [data-baseweb="menu"], [data-baseweb="popover"] li { background: var(--surface-2) !important; color: var(--ink) !important; }
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea { background: var(--surface-2) !important; color: var(--ink) !important; border-color: var(--border) !important; }
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { caret-color: var(--teal); }

    /* ====== DATAFRAME ====== */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface-2);
        color: var(--ink);
    }
    [data-testid="stDataFrame"] * { color: var(--ink); }

    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: var(--canvas);
        border-right: 1px solid var(--border);
    }

    /* ====== RADIO / CAPTION ====== */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label div { color: var(--ink) !important; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* ====== RESPONSIVE ====== */
    @media (max-width: 700px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        .app-header h1 { font-size: 1.75rem; }
        .fleet-cards { flex-direction: column; }
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
        [data-testid="stTabs"] button { padding: 0.6rem 0.4rem; font-size: 0.78rem; }
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown("""
<div class="app-header">
    <h1>🏥 MedEQ</h1>
    <p class="subtitle">Medical Equipment Predictive Maintenance</p>
    <p class="supporting">Explainable AI &bull; Clinical Triage &bull; AI Work Orders</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# AI PIPELINE VISUALISATION
# ===========================================================================



# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def call_api(endpoint: str, payload: dict, timeout: int = 10) -> dict | None:
    """
    Sends a POST request to the FastAPI backend and returns the JSON response.

    This is the single gateway between the Streamlit dashboard and the backend.
    Every API call (/predict, /explain, /dispatch) goes through here, so error
    handling and user-facing messages are centralized in one place rather than
    duplicated at every call site.

    PARAMETERS:
        endpoint: the API path (e.g. "predict", "explain", "dispatch").
        payload: dictionary of request body data, serialized as JSON.
        timeout: maximum seconds to wait for the backend to respond.

    RETURNS:
        Parsed JSON response as a dict, or None if the call failed. When it
        fails, a user-facing Streamlit error message is displayed inline so
        the operator knows exactly what went wrong and how to fix it.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend API is not running. Start it with: `uvicorn src.api:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("Backend API timed out. The model may be loading (or the LLM response is slow).")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend ({endpoint}). Error: {e}")
        return None


def check_system_status() -> tuple[bool, bool, str]:
    """Queries the backend to report backend + LLM state.

    The LLM check is a REAL probe against /test_llm (which auto-rotates
    through Gemini models when one exhausts its free-tier quota), cached
    in session_state for ~180s so we don't hammer the API (and the free
    per-day/per-minute quota) on every rerun — running an LLM probe on
    every button click is the fastest way to burn the free quota.

    Returns (backend_online, llm_available, message). This lets the
    dashboard surface WHY the SOP generator falls back to the offline
    template — the most common cause of a demo looking broken.
    """
    _now = time.time()
    cached = st.session_state.get("_llm_status_cache")
    if cached and (_now - cached[2]) < 180:
        return cached[0], cached[1], cached[3]

    backend_ok, llm_ok, msg = False, False, ""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=5)
        r.raise_for_status()
        backend_ok = True
        data = r.json()
        if not data.get("llm_available"):
            llm_ok, msg = False, data.get("llm_error") or "Backend online, but LLM is unavailable"
        else:
            # LLM configured — now verify it can actually answer (quota may
            # still be exhausted even though config succeeded).
            try:
                probe = requests.get(f"{API_BASE_URL}/test_llm", timeout=(5, 10))
                probe.raise_for_status()
                probe_data = probe.json()
                if probe_data.get("status") == "LLM is working":
                    llm_ok, msg = True, f"Gemini LLM active ({probe_data.get('model', '')})".strip()
                else:
                    err = re.sub(r"\s+", " ", str(probe_data.get("error", "")))
                    llm_ok, msg = False, f"LLM configured but calls are failing: {err}"
            except requests.exceptions.RequestException as exc:
                llm_ok, msg = False, f"LLM probe failed: {exc}"
    except requests.exceptions.RequestException as exc:
        llm_ok, msg = False, f"Backend unreachable: {exc}"

    st.session_state["_llm_status_cache"] = (backend_ok, llm_ok, _now, msg)
    return backend_ok, llm_ok, msg


def get_risk_level(risk_pct: float) -> str:
    """
    Maps a risk percentage to a severity tier label used for styling.

    Returns one of:
        - "critical" for risk >= 70% (action this week)
        - "warning"  for risk >= 40% (next scheduled maintenance window)
        - "routine"  for risk <  40% (continue monitoring)

    These thresholds are business rules aligned with hospital scheduling
    practices and are deliberately identical to the backend's thresholds
    (RISK_CRITICAL_THRESHOLD / RISK_WARNING_THRESHOLD in rul_triage.py) so
    the dashboard and API always agree on severity.
    """
    if risk_pct >= 70:
        return "critical"
    elif risk_pct >= 40:
        return "warning"
    return "routine"


def get_risk_label(risk_pct: float) -> str:
    """
    Maps a risk percentage to a short uppercase text label for badges.

    Returns "HIGH", "MEDIUM", or "LOW" using the same thresholds as
    get_risk_level(). Useful wherever a compact status word is needed
    (risk badges, table rows, RUL captions) instead of the tier slug.
    """
    if risk_pct >= 70:
        return "HIGH"
    elif risk_pct >= 40:
        return "MEDIUM"
    return "LOW"


def get_ward_badge_class(ward: str) -> str:
    """
    Maps a ward name to a CSS badge class for color-coding.

    Rule:
        - ward names containing "icu"      -> "icu"      (blue badge)
        - ward names containing "storage"  -> "storage"  (grey badge)
        - anything else                    -> "general"  (teal badge)

    Substring matching (case-insensitive) keeps the function robust to
    slightly different ward naming conventions ("ICU", "ICU Ward", etc.)
    without an exhaustive lookup table.
    """
    w = ward.lower()
    if "icu" in w:
        return "icu"
    if "storage" in w:
        return "storage"
    return "general"


def render_fleet_overview(alerts: list):
    """
    Renders the four summary cards at the top of the Device Risk List:
    total devices, and the count of high / medium / low risk devices.

    The counts are derived by classifying every alert through
    get_risk_level() — the same tier logic used everywhere else in the
    dashboard, so the summary always matches the detailed tables below.

    PARAMETERS:
        alerts: list of alert dicts returned by the backend /predict
                endpoint (each containing raw_risk_score_pct).
    """
    total = len(alerts)
    high = sum(1 for a in alerts if get_risk_level(a["raw_risk_score_pct"]) == "critical")
    med = sum(1 for a in alerts if get_risk_level(a["raw_risk_score_pct"]) == "warning")
    low = sum(1 for a in alerts if get_risk_level(a["raw_risk_score_pct"]) == "routine")

    st.markdown("""
    <div class="fleet-overview">
        <div class="fleet-label">Fleet Overview</div>
        <div class="fleet-cards">
    """, unsafe_allow_html=True)

    cards = [
        ("total", str(total), "Devices"),
        ("high", str(high), "High Risk"),
        ("med", str(med), "Medium Risk"),
        ("low", str(low), "Low Risk"),
    ]
    for cls, val, desc in cards:
        st.markdown(f"""
        <div class="fleet-card {cls}">
            <div class="fleet-value">{val}</div>
            <div class="fleet-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_rul_trajectory(rul_days: float, color: str) -> str:
    """Small inline SVG 'RUL trajectory' sparkline.

    Honest visualization of the countdown: remaining useful life declines
    linearly toward End-of-Life day. Shows the 'now' position (remaining
    days) and the EOL point, so the trajectory + value are both visible
    without inventing a fake degradation curve.
    """
    rul_days = max(0.0, float(rul_days))
    w, h = 320, 52
    x0, x1 = 24, 296
    y0, y1 = 10, 42
    # Sparkline runs from the EOL day back to today: left edge = today at
    # remaining life, right edge = EOL at 0 remaining.
    grad_id = f"rulg{int(rul_days) % 10000}"
    svg = f"""
    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%;">
        <defs>
            <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="{color}" stop-opacity="1"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.25"/>
            </linearGradient>
        </defs>
        <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="rgba(148,183,194,0.25)" stroke-width="1" stroke-dasharray="3 3"/>
        <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="url(#{grad_id})" stroke-width="2" stroke-linecap="round"/>
        <circle cx="{x0}" cy="{y0}" r="3.5" fill="{color}"/>
        <circle cx="{x1}" cy="{y1}" r="3" fill="rgba(148,183,194,0.6)"/>
        <text x="{x0}" y="22" font-size="9" fill="{color}" font-weight="700">{rul_days:.0f} days</text>
        <text x="{x0}" y="33" font-size="8" fill="rgba(148,183,194,0.85)">TODAY</text>
        <text x="{x1 - 22}" y="33" font-size="8" fill="rgba(148,183,194,0.85)" text-anchor="end">EOL</text>
        <text x="{x0}" y="{h - 3}" font-size="8" fill="rgba(148,183,194,0.7)">Active life remaining →</text>
        <text x="{x1}" y="{h - 3}" font-size="8" fill="rgba(148,183,194,0.7)" text-anchor="end">End of life</text>
    </svg>
    """
    return svg


def render_alert_card(alert: dict):
    """
    Renders one device's full HTML alert card: name, ward badge, risk
    percentage, triage priority, RUL countdown bar + trajectory sparkline,
    recommended action, and the top SHAP risk drivers.

    This is the primary per-device visualization used in both the Device
    Risk List details view and (via shared helpers) the What-If simulator.
    All numeric values come pre-computed from the backend alert dict, so
    this function is purely presentational.

    PARAMETERS:
        alert: an alert dict produced by the backend /predict endpoint
               (see build_priority_alert() in src/rul_triage.py for the
               exact schema).
    """
    risk_pct = alert["raw_risk_score_pct"]
    level = get_risk_level(risk_pct)
    ward_class = get_ward_badge_class(alert["ward_criticality"])

    # RUL bar: show remaining life as a fraction of max_days. Colored by
    # urgency so the countdown reads instantly (red = act now).
    rul_days = alert["estimated_days_remaining"]
    rul_max = 365
    rul_pct = min(100, max(2, float(rul_days) / rul_max * 100))
    rul_color = {"critical": "var(--high)", "warning": "var(--medium)", "routine": "var(--low)"}[level]
    rul_label = "Critical — act now" if level == "critical" else ("Scheduled window" if level == "warning" else "Routine monitoring")

    driver_html = ""
    max_impact = max((abs(d["impact"]) for d in alert.get("top_drivers", [])), default=1)
    for d in alert.get("top_drivers", []):
        impact = d["impact"]
        impact_class = "positive" if impact > 0 else "negative"
        arrow = "↑" if impact > 0 else "↓"
        bar_pct = min(100, (abs(impact) / max_impact) * 100) if max_impact > 0 else 0
        driver_html += f"""
        <div class="driver-row">
            <span class="driver-feature">{d['feature']}</span>
            <div style="flex:1;margin:0 12px;">
                <div class="driver-bar {impact_class}" style="width:{bar_pct}%;"></div>
            </div>
            <span class="driver-impact {impact_class}">{arrow} {impact:+.4f}</span>
        </div>
        """

    card_html = f"""
    <div class="alert-card {level}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
            <div style="flex:1;min-width:180px;">
                <div class="alert-device-name">{alert['device_name']}</div>
                <div class="alert-classification">{alert['classification']}</div>
                <span class="ward-badge {ward_class}">{alert['ward_criticality']}</span>
            </div>
            <div style="text-align:right;min-width:100px;">
                <div class="priority-score">{alert['priority_score']}</div>
                <div class="priority-label">Triage Priority</div>
            </div>
        </div>

        <div style="display:flex;gap:20px;margin:16px 0 12px;flex-wrap:wrap;align-items:center;">
            <div class="risk-badge {level}">
                <span class="risk-label">{get_risk_label(risk_pct)}</span>
                <span class="risk-pct">{risk_pct}%</span>
            </div>
            <div class="metric-block">
                <div class="metric-value">{alert['estimated_days_remaining']} <span style="font-size:0.8rem;font-weight:600;color:var(--muted);">days</span></div>
                <div class="metric-label">Remaining Useful Life (RUL)</div>
                <div class="rul-bar-wrap">
                    <div class="rul-bar-track">
                        <div class="rul-bar-fill" style="width:{rul_pct}%;background:{rul_color};"></div>
                    </div>
                    <div class="rul-bar-caption">{rul_label} — {rul_pct:.0f}% of 1-year life remaining</div>
                </div>
            </div>
        </div>

        <div style="margin:4px 0 2px;">
            {render_rul_trajectory(alert['estimated_days_remaining'], rul_color)}
            <div class="rul-bar-caption">RUL trajectory — remaining life declines toward End-of-Life (EOL) day. Re-inspect on any high-risk alert.</div>
        </div>

        <div class="action-badge {level}">{alert['recommended_action']}</div>

        <div style="margin-top:14px;">
            <div style="font-size:0.78rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                Top Risk Drivers
            </div>
            {driver_html}
        </div>
    </div>
    """
    st.html(card_html)


def _clean_sop_text(sop_text: str) -> str:
    """Strip control characters and decorative separator lines from raw
    LLM output.

    WHY THIS EXISTS (and why BOTH the on-screen renderer and the
    downloadable .txt go through it): Gemini occasionally inserts
    invisible control characters (e.g. \\x01) or digit/punctuation runs
    (e.g. "1/1/1", "1.1.1", "1-1-1") as decorative separators. If they
    reach the UI they render as visible garbage like "/1/1/1" or bold
    "\\1" repeats. This strips them once, in one place, before any HTML
    parsing.

    Handles:
      - C0/C1 control characters plus the Unicode BOM/zero-width chars
      - any line that is a long run of <=2 distinct characters
        ("=======", "------", "~~~", "**", "xxxxx")
      - digit-separator "counter" lines ("1/1/1", "1.1.1", "1-1-1",
        "1\\1\\1"), even when wrapped in markdown asterisks/ticks
    """
    sop_text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f\ufeff\u200b\u200c\u200d]",
        "", sop_text,
    )

    # If Gemini returned raw HTML instead of plain text, convert it back to
    # plain text so the line-by-line parser below works correctly.
    if re.search(r"<(?:div|p|span|ul|ol|li|strong|em|br|h[1-6])[\s>]", sop_text, re.IGNORECASE):
        # Convert HTML block/inline elements to plain text equivalents
        sop_text = re.sub(r"<br\s*/?>", "\n", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(?:div|p|li|h[1-6])>", "\n", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(?:div|p|ul|ol)[^>]*>", "", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<li[^>]*>", "- ", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(strong|b)>", "**", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(strong|b)>", "**", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(em|i)>", "*", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(em|i)>", "*", sop_text, flags=re.IGNORECASE)
        # Strip any remaining HTML tags
        sop_text = re.sub(r"<[^>]+>", "", sop_text)
        # Decode common HTML entities
        sop_text = sop_text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        sop_text = sop_text.replace("&nbsp;", " ").replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"')
    kept_lines = []
    for line in sop_text.split("\n"):
        # Remove markdown brace-markers so the *content* decides whether
        # this is a separator run (e.g. "** /1/1/1 **" -> "1/1/1").
        dec = re.sub(r"[#*_~`]", "", line).replace(" ", "")
        if len(dec) > 3 and len(set(dec)) <= 2:
            continue
        if dec and re.fullmatch(r"\d+(?:[./\\-]\d+)+", dec):
            continue

        # Remove INLINE digit-separator noise too — e.g. "(1/1/1)",
        # "/1/1/1" glued to trailing junk, or "…1/1/1" after a bullet.
        # A token like "1/1/1" repeats the SAME number group three times,
        # which no legitimate work-order content does, so backreferencing
        # the first group deletes the decorative tick-mark while leaving
        # real values ("1/2/3", a date, "3.1 V") untouched.
        line = re.sub(r"(?<![A-Za-z0-9])[./\\-]?(\d+)(?:[./\\-]\1){2,}", "", line)
        # Collapse leftover empty pairs and raw separator junk.
        line = re.sub(r"\(\)|\[\]|\{\}", "", line)
        line = re.sub(r"[~`^]{2,}", "", line)
        line = re.sub(r"\s{2,}", " ", line).rstrip()

        # Drop lines that are now empty or only punctuation.
        if not re.sub(r"[^A-Za-z0-9]", "", line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _esc_sop_text(text: str) -> str:
    """HTML-escape SOP text before embedding it in the card HTML.

    Guarantees no stray ``<``/``>``/``&`` that survived every cleaner can
    ever be interpreted as real markup or break the card layout — while
    still converting markdown ``**bold**`` to a styled ``<strong>``. We
    escape FIRST (neutralizing any leftover Gemini HTML), then apply our
    own safe bold conversion, so the rendered page is always clean.
    """
    text = html.escape(str(text), quote=False)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def render_sop_display(sop_text: str, source: str, device_name: str, llm_error: str | None = None, model: str | None = None):
    """Parse and render SOP as a clean MEDICAL WORK ORDER in pure Markdown.

    WHY PURE MARKDOWN (NO HTML AT ALL): SOP output used to be rendered as
    a custom HTML card via st.markdown(..., unsafe_allow_html=True). If
    the browser's markdown renderer ever fails to parse that HTML (or the
    raw Gemini HTML slips through), the literal "<div class=...>" tags
    show up AS TEXT on screen — exactly the recurring bug we chased.
    Rewriting the renderer to emit ONLY Markdown (headings, bold, bullets,
    emoji) removes the entire failure class: there is no HTML anywhere in
    this function's output, so raw tags can never appear. Streamlit's
    native markdown is bulletproof for plain text.
    """
    # Re-clean (idempotent) and guard against ANY entity-encoded leftovers.
    sop_text = _clean_sop_text(sop_text)
    sop_text = (sop_text.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'"))
    lines = sop_text.strip().split("\n")

    md_lines: list[str] = []

    # Header as plain markdown (no custom HTML classes needed).
    md_lines.append(f"### SOP Work Order — {device_name}")
    if source == "llm":
        md_lines.append(f"*Generated by Gemini LLM*" + (f" · {model}" if model else ""))
    else:
        md_lines.append("*Offline Template — LLM unavailable*")
    md_lines.append("")

    # Section titles are matched in BOTH raw and LLM-prefixed forms
    # ("Device Information", "### 1. Device Information", etc.) because
    # Gemini occasionally formats section headers slightly differently.
    _section_keywords = (
        "DEVICE INFORMATION", "RISK SUMMARY", "LIKELY ROOT CAUSES",
        "RECOMMENDED MAINTENANCE STEPS", "SAFETY CONSIDERATIONS",
        "REQUIRED INSPECTION", "URGENCY", "EXPECTED ACTION",
    )
    _section_titles = {
        "device information": "Device Information",
        "device info": "Device Information",
        "risk summary": "Risk Summary",
        "summary": "Risk Summary",
        "likely root causes": "Likely Root Causes",
        "root causes": "Likely Root Causes",
        "recommended maintenance steps": "Recommended Maintenance Steps",
        "recommended steps": "Recommended Maintenance Steps",
        "recommended maintenance steps:": "Recommended Maintenance Steps",
        "safety considerations": "Safety Considerations",
        "required inspection": "Required Inspection",
        "urgency level": "Urgency Level",
        "urgency": "Urgency Level",
        "expected action": "Expected Action",
    }

    def _find_section_title(line: str) -> str | None:
        """Returns the canonical section title for a line, or None."""
        stripped = line.strip()
        # Strip markdown bold/italic markers first ("**1. Risk Summary**")
        stripped = stripped.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        # Strip markdown header prefixes: ### 1. Risk Summary -> Risk Summary
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        # Strip leading numbering: "3. Likely Root Causes" -> "Likely Root Causes"
        if " " in stripped:
            first, rest = stripped.split(" ", 1)
            if first.rstrip(".").isdigit():
                stripped = rest.strip()
        elif stripped.rstrip(".").isdigit():
            return None
        low = stripped.lower().rstrip(":").strip()
        if low in _section_titles:
            return _section_titles[low]
        for kw in _section_keywords:
            if kw.lower() in low:
                return _section_titles.get(low, kw.title())
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Belt-and-braces: no residual HTML may survive into Markdown.
        if "<" in stripped or ">" in stripped:
            stripped = re.sub(r"<br\s*/?>", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"</?(strong|b|em|i|p|div|li|ul|ol|span)>", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"<[^>]+>", "", stripped)
            stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            continue

        if stripped.startswith("=" * 10):
            continue
        if "MAINTENANCE WORK ORDER" in stripped or "SOP WORK ORDER" in stripped:
            continue
        if "Generated:" in stripped:
            continue

        # Urgency special-case -> big text line (no HTML pill; pure markdown).
        if stripped.upper().startswith("URGENCY LEVEL:"):
            level_text = stripped.split(":", 1)[1].strip()
            md_lines.append("#### Urgency Level")
            if "HIGH" in level_text.upper():
                md_lines.append("**🔴 HIGH** — " + level_text)
            elif "MEDIUM" in level_text.upper():
                md_lines.append("**🟡 MEDIUM** — " + level_text)
            else:
                md_lines.append("**🟢 LOW** — " + level_text)
            md_lines.append("")
            continue

        # Standalone bold urgency word (Gemini renders "**High**" alone).
        _bold_only = stripped.replace("**", "").strip().upper()
        if _bold_only in ("HIGH", "MEDIUM", "LOW") and "**" in stripped:
            mark = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[_bold_only]
            md_lines.append(f"**{mark} {_bold_only}**")
            continue

        section_title = _find_section_title(stripped)
        if section_title:
            md_lines.append(f"#### {section_title}")
            md_lines.append("")
            continue

        # Bullets -> markdown list items (convert any stray bullet glyphs).
        if stripped.startswith("- ") or stripped.startswith("• ") or stripped.startswith("* "):
            bullet_text = stripped[2:] if (stripped.startswith("- ") or stripped.startswith("* ")) else stripped[1:].strip()
            md_lines.append(f"- {bullet_text}")
            continue

        if stripped[0:1].isdigit() and ". " in stripped:
            md_lines.append(stripped)
            continue

        md_lines.append(stripped)

    # Assemble one clean markdown document. There is deliberately NO HTML
    # in it — just headings, bold and lists, which Streamlit renders
    # natively and safely.
    body = "\n".join(md_lines).strip() + "\n"
    st.markdown(body)

    # Surface WHY the LLM fell back (quota, network) as a native widget if
    # it did — still markdown-level messaging, no raw-text risk.
    if source != "llm" and llm_error:
        brief = re.sub(r"\s+", " ", str(llm_error))[:220]
        st.warning(
            f"**Gemini LLM unavailable** — this SOP used the offline template. "
            f"Reason: {brief}"
        )


# ===========================================================================
# LOAD CONFIG
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config", "device_schema.json")) as f:
    schema = json.load(f)
schema.pop("_comment", None)
categories = list(schema.keys())


# ===========================================================================
# SYSTEM STATUS INDICATOR — surface backend & LLM health so a demo never
# silently falls back to the offline SOP template without explanation.
# ===========================================================================
backend_ok, llm_ok, status_msg = check_system_status()
if backend_ok and llm_ok:
    st.markdown(
        f'<div class="status-row">'
        f'<span class="status-pill online"><span class="dot"></span>System Online</span>'
        f'<span class="status-pill online"><span class="dot"></span>LLM Active — {status_msg}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
elif backend_ok:
    st.markdown(
        f'<div class="status-row">'
        f'<span class="status-pill online"><span class="dot"></span>System Online</span>'
        f'<span class="status-pill offline"><span class="dot"></span>LLM Offline</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        f"**Backend is online, but the Gemini LLM is unavailable** — SOP Work Orders "
        f"will use the offline template while it recovers. {status_msg}"
    )
else:
    st.error(
        f"**Backend API not reachable.** Start it with `uvicorn src.api:app --reload`. {status_msg}"
    )


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Device Risk List",
    "🎛️ What-If Simulator",
    "📊 Model Comparison",
    "⏳ RUL Estimator",
])


# =======================================================================
# TAB 1: DEVICE RISK LIST
# =======================================================================
with tab1:
    col_model, col_btn = st.columns([1, 1])
    with col_model:
        model_choice_1 = st.radio(
            "Model variant",
            ["realistic", "optimized"],
            horizontal=True,
            key="tab1_model",
            help="realistic: ~95% accuracy, real-world label randomness. optimized: ~99% accuracy, deterministic label.",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_scan = st.button("🔍 Run Risk Scan", type="primary", width="stretch")

    # -------------------------------------------------------------------
    # RUN RISK SCAN: sample 50 devices from the synthetic telemetry file,
    # POST each one to the backend /predict endpoint, and collect the
    # resulting alert objects. Results are stored in session_state so the
    # table/detail views below can render them on every rerun without
    # re-scoring the fleet.
    # -------------------------------------------------------------------
    if run_scan:
        with st.spinner("Scanning fleet..."):
            data_path = os.path.join(BASE_DIR, "data", "synthetic_telemetry_realistic.csv")
            # Fixed random seed keeps the demo reproducible — the same 50
            # devices appear every time the button is clicked.
            df = pd.read_csv(data_path).sample(50, random_state=1)

            alerts = []
            progress = st.progress(0, text="Scoring devices...")
            for i, (_, row) in enumerate(df.iterrows()):
                # Build the payload exactly in the format the backend
                # TelemetryInput model expects (see src/api.py).
                category = row["classification"]
                ward = schema[category]["ward_criticality"]
                payload = {
                    "device_name": row["device_name"],
                    "classification": category,
                    "ward_criticality": ward,
                    "age_fraction": row["age_fraction"],
                    "temperature": row.get("temperature", 0.0) if not pd.isna(row.get("temperature", np.nan)) else 0.0,
                    "vibration": row.get("vibration", 0.0) if not pd.isna(row.get("vibration", np.nan)) else 0.0,
                    "voltage": row.get("voltage", 0.0) if not pd.isna(row.get("voltage", np.nan)) else 0.0,
                    "hours_used": row.get("hours_used", 0.0) if not pd.isna(row.get("hours_used", np.nan)) else 0.0,
                    "manufacturer_risk_tier": row.get("manufacturer_risk_tier", "low"),
                    "model_variant": model_choice_1,
                }
                result = call_api("predict", payload)
                if result:
                    # Attach only the telemetry fields this device's category
                    # genuinely tracks (matches device_schema.json), so the
                    # SOP generator never sees fabricated zeros.
                    result["telemetry"] = {
                        field: payload[field]
                        for field in ["temperature", "vibration", "voltage", "hours_used"]
                        if field in schema[category]["fields"]
                    }
                    result["manufacturer_risk_tier"] = payload["manufacturer_risk_tier"]
                    alerts.append(result)
                progress.progress((i + 1) / len(df), text=f"Scoring device {i + 1}/{len(df)}...")
            progress.empty()

            # Persist results; clear any stale SOP from a previous scan so
            # the detail view never shows a work order for a device that is
            # no longer selected.
            st.session_state["risk_scan_alerts"] = alerts
            st.session_state.pop("sop_result", None)
            st.session_state.pop("sop_device_name", None)

    alerts = st.session_state.get("risk_scan_alerts", [])

    if alerts:
        alerts_sorted = sorted(alerts, key=lambda a: -a["priority_score"])

        render_fleet_overview(alerts_sorted)

        st.markdown('<div class="section-header">Device Risk Priority</div>', unsafe_allow_html=True)

        # Paginated table
        page_size = 10
        page_count = max(1, (len(alerts_sorted) + page_size - 1) // page_size)
        page_options = [f"Page {p} of {page_count}" for p in range(1, page_count + 1)]

        col_page, col_hint = st.columns([1, 2])
        with col_page:
            selected_page = st.selectbox("Navigate", page_options, label_visibility="collapsed", key="risk_results_page")
        page_number = page_options.index(selected_page) + 1
        start = (page_number - 1) * page_size
        page_alerts = alerts_sorted[start:start + page_size]

        table_rows = []
        for alert in page_alerts:
            risk = alert["raw_risk_score_pct"]
            table_rows.append({
                "Device": alert["device_name"],
                "Category": alert["classification"],
                "Ward": alert["ward_criticality"],
                "Risk": f"{risk:.1f}%",
                "Status": get_risk_label(risk),
                "Priority": f"{alert['priority_score']:.1f}",
                "RUL (days)": f"{alert['estimated_days_remaining']:.1f}",
            })

        with col_hint:
            st.caption(f"Showing {start + 1}-{min(start + page_size, len(alerts_sorted))} of {len(alerts_sorted)} devices, sorted by triage priority")

        # Click a table row to select that device — the detail view below
        # (RUL bar + risk drivers + SHAP) follows the table selection.
        selection = st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            width="stretch",
            height=430,
            on_select="rerun",
            selection_mode="single-row",
            key="risk_results_table",
        )

        # Determine which device is "selected": prefer the row clicked in
        # the table; otherwise fall back to the dropdown.
        detail_options = [f"{a['device_name']} | {a['raw_risk_score_pct']:.1f}% risk" for a in page_alerts]
        try:
            selected_rows = st.session_state["risk_results_table"]["selection"]["rows"]
            table_idx = selected_rows[0] if selected_rows else None
        except (KeyError, IndexError, TypeError):
            table_idx = None

        if table_idx is not None and 0 <= table_idx < len(page_alerts):
            selected_alert = page_alerts[table_idx]
            st.success(f"Row selected: **{selected_alert['device_name']}** — details shown below. Click a different row or use the dropdown to change.")
        else:
            selected_detail = st.selectbox("View device details", detail_options, key="risk_detail_device")
            selected_alert = page_alerts[detail_options.index(selected_detail)]

        st.markdown('<div class="section-header">Selected Device</div>', unsafe_allow_html=True)
        render_alert_card(selected_alert)

        # SHAP Explanation
        st.markdown('<div class="section-header">Why This Risk? (SHAP Explanation)</div>', unsafe_allow_html=True)
        max_impact = max((abs(d["impact"]) for d in selected_alert.get("top_drivers", [])), default=1)
        for d in selected_alert.get("top_drivers", []):
            impact = d["impact"]
            color = "var(--high)" if impact > 0 else "var(--low)"
            direction = "increases" if impact > 0 else "decreases"
            bar_w = min(100, (abs(impact) / max_impact) * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <div style="min-width:180px;font-weight:600;font-size:0.9rem;">{d['feature']}</div>
<div style="flex:1;background:var(--track);border-radius:4px;height:10px;">
                        <div style="width:{bar_w}%;height:100%;background:{color};border-radius:4px;"></div>
                    </div>
                    <div style="min-width:90px;font-weight:700;color:{color};text-align:right;">{impact:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Red bars = feature increases predicted risk. Green bars = feature decreases predicted risk.")

        # SOP Generation
        st.markdown('<div class="section-header">Generate SOP Work Order</div>', unsafe_allow_html=True)
        sop_options = [
            f"{a['device_name']} | {a['classification']} | {a['raw_risk_score_pct']}% risk"
            for a in alerts_sorted
        ]
        selected_option = st.selectbox("Select device for work order", sop_options, index=None, placeholder="Choose a device", key="sop_device_select")

        sop_alert = None
        if selected_option is not None:
            sop_idx = sop_options.index(selected_option)
            sop_alert = alerts_sorted[sop_idx]
            lvl = get_risk_level(sop_alert["raw_risk_score_pct"])
            st.info(f"**{sop_alert['device_name']}** — {sop_alert['raw_risk_score_pct']}% risk ({sop_alert['classification']}) — ~{sop_alert['estimated_days_remaining']} days remaining")

        if st.button("📝 Generate SOP Work Order", disabled=sop_alert is None, type="primary"):
            with st.spinner("Generating SOP..."):
                sop_result = call_api("dispatch", {
                    "device_name": sop_alert["device_name"],
                    "classification": sop_alert["classification"],
                    "risk_score_pct": sop_alert["raw_risk_score_pct"],
                    "top_drivers": sop_alert["top_drivers"],
                    "telemetry": sop_alert.get("telemetry", {}),
                    "ward_criticality": sop_alert["ward_criticality"],
                    "estimated_days_remaining": sop_alert["estimated_days_remaining"],
                    "recommended_action": sop_alert["recommended_action"],
                }, timeout=90)
                if sop_result:
                    # BELT-AND-SUSPENDERS: the backend already sanitizes LLM
                    # output, but Normalize it AGAIN here the moment the
                    # response arrives. That way every consumer of
                    # st.session_state["sop_result"] (on-screen render,
                    # .txt download, any future feature) is guaranteed to
                    # see clean plain text — even if a future code path
                    # bypasses/forgets the cleaner. Never store raw LLM
                    # output in the session: raw HTML from Gemini is the
                    # exact bug we've been chasing.
                    sop_result["sop"] = _clean_sop_text(str(sop_result.get("sop", "")))
                    st.session_state["sop_result"] = sop_result
                    st.session_state["sop_device_name"] = sop_alert["device_name"]

        saved_sop = st.session_state.get("sop_result")
        if saved_sop:
            render_sop_display(
                saved_sop["sop"],
                saved_sop["source"],
                st.session_state["sop_device_name"],
                llm_error=saved_sop.get("llm_error"),
                model=saved_sop.get("model"),
            )

            col_dl, col_clear = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    "📥 Download SOP",
                    _clean_sop_text(saved_sop["sop"]),
                    file_name=f"SOP_{st.session_state['sop_device_name'].replace(' ', '_')}.txt",
                    mime="text/plain",
                )
            with col_clear:
                if st.button("🗑️ Clear SOP"):
                    st.session_state.pop("sop_result", None)
                    st.session_state.pop("sop_device_name", None)
                    st.rerun()


# =======================================================================
# TAB 2: WHAT-IF SIMULATOR
# -----------------------------------------------------------------------
# Live "what if" exploration: the user adjusts device configuration and
# sensor sliders, and a prediction is fetched from the backend in real
# time. This demonstrates the explainability + interactivity of the
# pipeline without needing a full fleet scan.
# =======================================================================
with tab2:
    col_left, col_right = st.columns(2)

    # Left column: device-level configuration (category, age, manufacturer
    # recall tier, model variant). Ward criticality comes straight from the
    # device schema so the simulator always uses realistic ward mappings.
    with col_left:
        st.markdown("##### Device Configuration")
        sim_category = st.selectbox("Device Category", categories, key="sim_category")
        sim_ward = schema[sim_category]["ward_criticality"]
        st.caption(f"Ward criticality: **{sim_ward}**")

        sim_age = st.slider("Age Fraction (0 = new, 1 = end of life)", 0.0, 1.0, 0.5, 0.01, key="sim_age")
        sim_manufacturer_tier = st.select_slider("Manufacturer Recall History", options=["low", "medium", "high"], value="low", key="sim_tier")

        model_choice_2 = st.radio("Model variant", ["realistic", "optimized"], horizontal=True, key="tab2_model")

    # Right column: one slider per telemetry field the selected category
    # actually tracks (from device_schema.json). Fields the category does
    # NOT track are forced to 0.0 — identical to how training data treated
    # them — and the user is told they're not tracked.
    with col_right:
        st.markdown("##### Sensor Readings")
        fields = schema[sim_category]["fields"]
        ranges = schema[sim_category]["ranges"]

        sim_values = {}
        for field in ["temperature", "vibration", "voltage", "hours_used"]:
            if field in fields:
                low, high = ranges[field]
                sim_values[field] = st.slider(field.replace("_", " ").title(), float(low), float(high), float((low + high) / 2), key=f"sim_{field}")
            else:
                sim_values[field] = 0.0

        missing_fields = [f for f in ["temperature", "vibration", "voltage", "hours_used"] if f not in fields]
        if missing_fields:
            st.caption(f"{', '.join(f.replace('_',' ').title() for f in missing_fields)} not tracked for {sim_category}")

    # "Calculate Risk" -> assemble the payload from every slider and POST
    # to /predict. The simulated device has a placeholder name; the results
    # (risk %, RUL, priority, SHAP drivers) render in a styled card below.
    if st.button("⚡ Calculate Risk →", type="primary", width="stretch"):
        with st.spinner("Running prediction..."):
            payload = {
                "device_name": "What-If Simulated Device",
                "classification": sim_category,
                "ward_criticality": sim_ward,
                "age_fraction": sim_age,
                "manufacturer_risk_tier": sim_manufacturer_tier,
                "model_variant": model_choice_2,
                **sim_values,
            }
            result = call_api("predict", payload)

        if result:
            risk = result["raw_risk_score_pct"]
            level = get_risk_level(risk)

            risk_color = {"critical": "var(--high)", "warning": "var(--medium)", "routine": "var(--low)"}[level]
            risk_icon = {"critical": "🔴", "warning": "🟡", "routine": "🟢"}[level]

            rul_days = result["estimated_days_remaining"]
            rul_pct = min(100, max(2, float(rul_days) / 365 * 100))
            rul_label = {"critical": "Critical — act now", "warning": "Scheduled window", "routine": "Routine monitoring"}[level]

            st.markdown(f"""
            <div class="whatif-result">
                <div style="text-align:center;margin-bottom:8px;">
                    <div class="whatif-risk-label">Predicted Failure Risk</div>
                </div>
                <div style="text-align:center;">
                    <div class="whatif-risk-large" style="color:{risk_color};">{risk_icon} {get_risk_label(risk)} FAILURE RISK</div>
                    <div class="whatif-risk-large" style="color:{risk_color};font-size:2.5rem;margin-top:4px;">{risk}%</div>
                </div>
                <div style="display:flex;justify-content:center;gap:48px;margin-top:20px;flex-wrap:wrap;">
                    <div style="text-align:center;">
                        <div style="font-size:1.6rem;font-weight:700;color:var(--teal);">{rul_days} days</div>
                        <div class="metric-label">Remaining Useful Life (RUL)</div>
                        <div class="rul-bar-wrap" style="margin:10px auto 0;text-align:left;">
                            <div class="rul-bar-track">
                                <div class="rul-bar-fill" style="width:{rul_pct}%;background:{risk_color};"></div>
                            </div>
                            <div class="rul-bar-caption">{rul_label} — {rul_pct:.0f}% of 1-year life remaining; declines toward EOL daily</div>
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div class="priority-score" style="font-size:1.6rem;">{result['priority_score']}</div>
                        <div class="priority-label">Triage Priority</div>
                    </div>
                    <div style="text-align:center;">
                        <span class="ward-badge {get_ward_badge_class(sim_ward)}" style="font-size:0.85rem;padding:5px 14px;">{sim_ward}</span>
                        <div class="metric-label" style="margin-top:4px;">Ward</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="action-badge {level}" style="margin-top:14px;font-size:0.92rem;">
                {result['recommended_action']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Why This Score? (SHAP Feature Drivers)</div>', unsafe_allow_html=True)

            drivers = result["top_drivers"]
            max_imp = max((abs(d["impact"]) for d in drivers), default=1)
            for d in drivers:
                imp = d["impact"]
                color = "var(--high)" if imp > 0 else "var(--low)"
                bar_w = min(100, (abs(imp) / max_imp) * 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <div style="min-width:180px;font-weight:600;font-size:0.9rem;">{d['feature']}</div>
                    <div style="flex:1;background:var(--track);border-radius:4px;height:10px;">
                        <div style="width:{bar_w}%;height:100%;background:{color};border-radius:4px;"></div>
                    </div>
                    <div style="min-width:90px;font-weight:700;color:{color};text-align:right;">{imp:+.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Red = increases risk. Green = decreases risk.")


# =======================================================================
# TAB 3: MODEL COMPARISON
# =======================================================================
with tab3:
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:1.15rem;font-weight:700;color:var(--ink);margin-bottom:4px;">
            Realistic vs. Optimized Model — Full Transparency
        </div>
        <div style="color:var(--muted);font-size:0.92rem;">
            Two model variants trained on purpose, showing the honest tradeoff rather than a single number without context.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        metrics = requests.get(f"{API_BASE_URL}/model_metrics", timeout=10).json()

        col1, col2 = st.columns(2)

        with col1:
            m = metrics["realistic"]["classifier"]
            st.markdown(f"""
            <div class="model-card realistic">
                <div class="model-title">Realistic Model</div>
                <div style="color:var(--muted);font-size:0.85rem;margin-bottom:16px;">
                    15% of training labels include real-world randomness.
                </div>
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="mt-value">{m['accuracy']*100:.1f}%</div>
                        <div class="mt-label">Accuracy</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['precision']*100:.1f}%</div>
                        <div class="mt-label">Precision</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['recall']*100:.1f}%</div>
                        <div class="mt-label">Recall</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['f1']:.3f}</div>
                        <div class="mt-label">F1 Score</div>
                    </div>
                </div>
                <div class="action-badge routine" style="margin-top:12px;">Recommended for production — honest accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            m = metrics["optimized"]["classifier"]
            st.markdown(f"""
            <div class="model-card optimized">
                <div class="model-title">Optimized Model</div>
                <div style="color:var(--muted);font-size:0.85rem;margin-bottom:16px;">
                    Fully deterministic label — failure = 1 when computed risk crosses 50%.
                </div>
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="mt-value">{m['accuracy']*100:.1f}%</div>
                        <div class="mt-label">Accuracy</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['precision']*100:.1f}%</div>
                        <div class="mt-label">Precision</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['recall']*100:.1f}%</div>
                        <div class="mt-label">Recall</div>
                    </div>
                    <div class="metric-tile">
                        <div class="mt-value">{m['f1']:.3f}</div>
                        <div class="mt-label">F1 Score</div>
                    </div>
                </div>
                <div class="action-badge warning" style="margin-top:12px;">Higher accuracy, but label is a direct function of features</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-top:16px;">
            <div style="font-weight:700;color:var(--ink);margin-bottom:8px;">Why two models, shown honestly side by side:</div>
            <div style="color:var(--ink);font-size:0.92rem;line-height:1.7;">
                Since this is synthetic data, we control both the input features AND the failure label — which means very high accuracy is achievable, but <em>how</em> you achieve it matters:<br><br>
                <strong style="color:var(--teal);">Realistic model</strong>: keeps a small amount of intentional randomness in the label (15% of rows), so the classes aren't perfectly separable — this mirrors real-world unpredictability, and the resulting ~95% accuracy is a genuinely earned score.<br><br>
                <strong style="color:var(--navy);">Optimized model</strong>: uses a deterministic label directly derived from the same features the model sees, making the classes almost perfectly separable — the ~99% accuracy reflects how cleanly the label was defined, not real predictive difficulty.<br><br>
                Both are legitimate on synthetic data — we chose to show both rather than picking one number.
            </div>
        </div>
        """, unsafe_allow_html=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend API. Is it running?\n\nError: {e}")


# =======================================================================
# TAB 4: RUL ESTIMATOR — Remaining Useful Life, always visible & usable
# =======================================================================
with tab4:
    # src/ is already on sys.path (added at the top of this file), so we
    # import the exact RUL logic used by the backend — a single source of
    # truth rather than a duplicated formula in the dashboard.
    from rul_triage import estimate_rul_days

    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:1.15rem;font-weight:700;color:var(--ink);margin-bottom:4px;">
            ⏳ Remaining Useful Life (RUL) Estimator
        </div>
        <div style="color:var(--muted);font-size:0.92rem;">
            Converts a predicted failure probability into an estimated days-until-failure countdown.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns(2)

    with col_in:
        failure_pct = st.slider(
            "Predicted failure probability (%)",
            0.0, 100.0, 50.0, 1.0, key="rul_prob",
            help="0 = healthy device, 100 = imminent failure.",
        )
        max_life = st.select_slider(
            "Max expected service life (days)",
            options=[90, 180, 365, 730, 1095, 1825, 3650],
            value=365,
            key="rul_maxlife",
            help="The 'healthy device' endpoint — how long a fully healthy unit is expected to last.",
        )
        min_life = 1  # never count a device below this

        p = failure_pct / 100.0
        rul_days = estimate_rul_days(p, max_days=max_life, min_days=min_life)

        st.caption(
            f"**days = max_life × (1 − failure_probability)** — clamping the "
            f"probability to [0.01, 0.99] first, then flooring the result at 1 day. "
            f"At {failure_pct:.0f}% risk on a {max_life}-day life, that gives ~{rul_days} days."
        )
        st.info(
            "The XGBoost model predicts the failure_probability itself; this tab "
            "re-expresses that same number in days using a simple inverse formula — "
            "no separate 'failure date' model is needed."
        )

    with col_out:
        level = get_risk_level(failure_pct)
        rul_color = {"critical": "var(--high)", "warning": "var(--medium)", "routine": "var(--low)"}[level]
        rul_pct = min(100, max(2, float(rul_days) / max_life * 100))
        rul_label = {"critical": "Critical — act now", "warning": "Scheduled window", "routine": "Routine monitoring"}[level]

        st.html(f"""
        <div class="whatif-result" style="margin-top:0;">
            <div style="display:flex;align-items:baseline;justify-content:center;gap:10px;">
                <div class="whatif-risk-large" style="color:{rul_color};font-size:3.5rem;">{rul_days}</div>
                <span style="color:var(--muted);font-size:1.1rem;font-weight:600;">days</span>
            </div>
            <div class="whatif-risk-label" style="text-align:center;">Estimated Remaining Useful Life</div>
            <div class="rul-bar-wrap" style="margin:14px auto 0;max-width:280px;text-align:left;">
                <div class="rul-bar-track">
                    <div class="rul-bar-fill" style="width:{rul_pct}%;background:{rul_color};"></div>
                </div>
                <div class="rul-bar-caption">{rul_label} — {rul_pct:.0f}% of {max_life}-day life remaining</div>
            </div>
            <div style="margin-top:14px;text-align:center;">
                {render_rul_trajectory(rul_days, rul_color)}
            </div>
        </div>
        """)

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-top:16px;">
        <div style="font-weight:700;color:var(--ink);margin-bottom:8px;">
            Why is this a simple formula and not a separately trained model?
        </div>
        <div style="color:var(--ink);font-size:0.92rem;line-height:1.7;">
            We already trained a regression model to predict the
            <strong style="color:var(--teal);">failure_probability</strong> itself — that's where the real machine
            learning happens. This estimator just re-expresses that same number in a human-readable unit (days)
            using a straightforward inverse relationship. Building a second, separate "days until failure" model
            would require real failure-date timestamp data we don't have (the dataset is a recall/safety registry,
            not a time-series monitoring log), so it would be no more accurate than this transparent formula — but
            far harder for a hospital team to trust and audit. A visible rule wins here.
        </div>
    </div>
    """, unsafe_allow_html=True)
