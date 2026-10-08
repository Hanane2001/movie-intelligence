import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_FEATURE = ROOT / "data" / "features" / "movies_feature.csv"
DATA_CLUSTERED = ROOT / "data" / "features" / "movies_clustered.csv"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


ICONS = {
    "film": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>',
    "star": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
    "flame": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "crown": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 4 3 12h14l3-12-6 7-4-7-4 7-6-7zm3 16h14"/></svg>',
    "grid": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "chart": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
    "award": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/></svg>',
    "trending": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "play": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
}



GLOBAL_CSS = """
<style>
    :root {
        --primary: #E50914;
        --primary-glow: rgba(229, 9, 20, 0.45);
        --gold: #FFD700;
        --gold-soft: rgba(255, 215, 0, 0.15);
        --bg-base: #0a0c14;
        --bg-card: rgba(26, 29, 41, 0.65);
        --bg-card-solid: #1a1d29;
        --border: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.14);
        --text: #fafafa;
        --text-muted: #8b8d97;
        --success: #10b981;
        --info: #3b82f6;
        --warning: #f59e0b;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 20% -10%, rgba(229,9,20,0.10), transparent 60%),
            radial-gradient(900px 500px at 100% 10%, rgba(59,130,246,0.08), transparent 60%),
            linear-gradient(180deg, #0a0c14 0%, #0e1120 100%);
        background-attachment: fixed;
    }

    /* ---------- Animations ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.96); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-24px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(229,9,20,0.3); }
        50%      { box-shadow: 0 0 0 12px rgba(229,9,20,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -800px 0; }
        100% { background-position: 800px 0; }
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        background: linear-gradient(135deg,
            rgba(229,9,20,0.12) 0%,
            rgba(20,22,30,0.6) 40%,
            rgba(59,130,246,0.08) 100%);
        background-size: 220% 220%;
        animation: gradientShift 12s ease infinite;
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-lg);
        padding: 44px 48px;
        margin-bottom: 32px;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(600px 200px at 90% 20%, rgba(255,215,0,0.08), transparent 70%),
            radial-gradient(400px 200px at 10% 80%, rgba(229,9,20,0.12), transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        color: var(--text);
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, #ffffff 30%, #ffd7d7 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: var(--text-muted);
        font-size: 15px;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,215,0,0.10);
        border: 1px solid rgba(255,215,0,0.25);
        color: var(--gold);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* ---------- KPI Card (premium) ---------- */
    .kpi-card {
        position: relative;
        background: linear-gradient(145deg, rgba(30,33,45,0.9), rgba(20,22,30,0.9));
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 26px 24px;
        height: 160px;
        overflow: hidden;
        backdrop-filter: blur(14px);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: fadeInUp 0.7s ease-out backwards;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: -50%; right: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(229,9,20,0.08), transparent 40%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .kpi-card:hover {
        transform: translateY(-6px);
        border-color: rgba(229,9,20,0.4);
        box-shadow: 0 20px 40px -12px rgba(229,9,20,0.25);
    }
    .kpi-card:hover::before { opacity: 1; }

    .kpi-icon-wrap {
        position: absolute;
        top: 20px; right: 20px;
        width: 42px; height: 42px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        color: var(--primary);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .kpi-card:hover .kpi-icon-wrap {
        background: rgba(229,9,20,0.12);
        border-color: rgba(229,9,20,0.4);
        transform: scale(1.1) rotate(-6deg);
        box-shadow: 0 0 20px rgba(229,9,20,0.3);
    }
    .kpi-label {
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .kpi-value {
        color: var(--text);
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.8px;
    }
    .kpi-sub {
        color: var(--text-muted);
        font-size: 12px;
        margin-top: 8px;
        font-weight: 500;
    }
    .kpi-trend {
        position: absolute;
        bottom: 20px; right: 20px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
    }
    .kpi-trend.up   { background: rgba(16,185,129,0.12); color: var(--success); }
    .kpi-trend.info { background: rgba(59,130,246,0.12); color: var(--info); }
    .kpi-trend.gold { background: rgba(255,215,0,0.12); color: var(--gold); }

    /* ---------- Section headers ---------- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 20px 0;
        font-size: 18px;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.3px;
    }
    .section-header svg {
        color: var(--primary);
        width: 20px; height: 20px;
    }

    /* ---------- Chart Card ---------- */
    .chart-card {
        background: linear-gradient(145deg, rgba(30,33,45,0.85), rgba(20,22,30,0.85));
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 22px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(14px);
        animation: fadeInScale 0.6s ease-out backwards;
        transition: border-color 0.3s ease;
    }
    .chart-card:hover { border-color: var(--border-strong); }
    .chart-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.2px;
    }
    .chart-title svg {
        color: var(--primary);
        width: 18px; height: 18px;
    }

    /* ---------- Movie Card ---------- */
    .movie-card {
        background: linear-gradient(145deg, rgba(30,33,45,0.85), rgba(20,22,30,0.85));
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 18px 22px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 18px;
        backdrop-filter: blur(14px);
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInLeft 0.5s ease-out backwards;
    }
    .movie-card:hover {
        transform: translateX(6px);
        border-color: rgba(229,9,20,0.4);
        box-shadow: 0 12px 32px -8px rgba(229,9,20,0.25);
    }
    .movie-rank {
        font-size: 22px;
        font-weight: 800;
        color: var(--primary);
        min-width: 42px;
        text-align: center;
        background: rgba(229,9,20,0.08);
        border: 1px solid rgba(229,9,20,0.2);
        border-radius: 10px;
        padding: 8px 0;
        letter-spacing: -1px;
    }
    .movie-info { flex: 1; }
    .movie-title {
        color: var(--text);
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .movie-meta {
        color: var(--text-muted);
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .movie-score {
        background: linear-gradient(135deg, var(--primary), #b00710);
        color: white;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        min-width: 72px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(229,9,20,0.3);
        letter-spacing: 0.3px;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid transparent;
        letter-spacing: 0.2px;
    }
    .badge-info    { background: rgba(59,130,246,0.12); color: #60a5fa; border-color: rgba(59,130,246,0.2); }
    .badge-success { background: rgba(16,185,129,0.12); color: #34d399; border-color: rgba(16,185,129,0.2); }
    .badge-warning { background: rgba(245,158,11,0.12); color: #fbbf24; border-color: rgba(245,158,11,0.2); }
    .badge-gold    { background: rgba(255,215,0,0.12); color: var(--gold); border-color: rgba(255,215,0,0.25); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1120 0%, #0a0c14 100%);
        border-right: 1px solid var(--border);
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), #b00710);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 26px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 4px 14px rgba(229,9,20,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(229,9,20,0.45);
        animation: pulseGlow 1.6s infinite;
    }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def _parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return []


@st.cache_data(show_spinner=False)
def load_features():
    if not DATA_FEATURE.exists():
        return None
    df = pd.read_csv(DATA_FEATURE)
    for col in ["genres", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_clustered():
    if not DATA_CLUSTERED.exists():
        return None
    return pd.read_csv(DATA_CLUSTERED)


@st.cache_resource(show_spinner=False)
def load_tfidf_reco():
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
    from nlp.tfidf import prepare_text, create_tfidf

    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    sim_path = MODELS_DIR / "similarity_matrix.joblib"
    movies_path = MODELS_DIR / "movies_for_reco.csv"

    if vec_path.exists() and sim_path.exists() and movies_path.exists():
        vectorizer = joblib.load(vec_path)
        similarity_matrix = joblib.load(sim_path)
        df_reco = pd.read_csv(movies_path)
        df_reco["genres"] = df_reco["genres"].apply(_parse_list)
        return df_reco, vectorizer, similarity_matrix

    df = load_features()
    if df is None:
        return None, None, None
    df = prepare_text(df)
    tfidf_matrix, vectorizer = create_tfidf(
        df, max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return df.reset_index(drop=True), vectorizer, similarity_matrix


def kpi_card_html(label, value, sub, icon_key="film", trend=None, trend_type="info"):
    """Retourne le HTML d'une KPI card premium, avec icône SVG."""
    icon_svg = ICONS.get(icon_key, ICONS["film"])

    trend_html = ""
    if trend:
        trend_html = f'<div class="kpi-trend {trend_type}">{trend}</div>'

    return f"""
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{icon_svg}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {trend_html}
    </div>
    """


def render_kpis(kpis):
    """Affiche N KPI cards proprement alignées via st.columns."""
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.markdown(kpi, unsafe_allow_html=True)


def section_header(icon_key, text):
    icon_svg = ICONS.get(icon_key, ICONS["chart"])
    st.markdown(
        f'<div class="section-header">{icon_svg}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def movie_card_html(rank, title, genres, year, score):
    badges = "".join(
        f'<span class="badge badge-info">{g}</span>' for g in genres[:3]
    )
    return f"""
    <div class="movie-card">
        <div class="movie-rank">#{rank}</div>
        <div class="movie-info">
            <div class="movie-title">{title}</div>
            <div class="movie-meta">{badges}<span style="color:#4a4d5a;">•</span><span>{year}</span></div>
        </div>
        <div class="movie-score">{score:.1%}</div>
    </div>
    """


CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"

CHART_THEME = {
    "text": "#8b8d97",
    "grid": "rgba(255,255,255,0.05)",
    "tooltip_bg": "rgba(20, 22, 30, 0.98)",
    "tooltip_border": "#E50914",
}


def render_chart(container_id, chart_type, labels, data, chart_title,
                 color="#E50914", horizontal=False, multi_colors=False):
    palette = ["#E50914", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
               "#ec4899", "#06b6d4", "#f97316"]
    bg = palette[:len(labels)] if multi_colors else color

    data_json = json.dumps({
        "labels": labels,
        "datasets": [{
            "label": chart_title,
            "data": data,
            "backgroundColor": bg,
            "borderColor": bg,
            "borderWidth": 2,
            "borderRadius": 8,
            "tension": 0.4,
            "fill": chart_type == "line",
            "pointBackgroundColor": color if chart_type == "line" else None,
            "pointRadius": 3,
            "pointHoverRadius": 6,
        }],
    })

    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "indexAxis": "y" if horizontal else "x",
        "animation": {"duration": 1200, "easing": "easeOutQuart"},
        "plugins": {
            "legend": {"display": False},
            "tooltip": {
                "backgroundColor": CHART_THEME["tooltip_bg"],
                "titleColor": "#fafafa",
                "bodyColor": "#fafafa",
                "borderColor": CHART_THEME["tooltip_border"],
                "borderWidth": 1,
                "padding": 12,
                "cornerRadius": 10,
                "displayColors": False,
                "titleFont": {"size": 13, "weight": "bold"},
            },
        },
        "scales": {
            "x": {"ticks": {"color": CHART_THEME["text"], "font": {"size": 11}},
                  "grid": {"color": CHART_THEME["grid"], "drawBorder": False}},
            "y": {"ticks": {"color": CHART_THEME["text"], "font": {"size": 11}},
                  "grid": {"color": CHART_THEME["grid"], "drawBorder": False},
                  "beginAtZero": True},
        },
    }

    html = f"""
    <div class="chart-card">
        <div class="chart-title">{ICONS['chart']}<span>{chart_title}</span></div>
        <div style="position: relative; height: 280px;">
            <canvas id="{container_id}"></canvas>
        </div>
    </div>
    <script src="{CHARTJS_CDN}"></script>
    <script>
        (function() {{
            const el = document.getElementById('{container_id}');
            if (!el) return;
            new Chart(el, {{
                type: '{chart_type}',
                data: {data_json},
                options: {json.dumps(options)},
            }});
        }})();
    </script>
    """
    st.components.v1.html(html, height=360, scrolling=False)


st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 24px 0 16px 0;">
        <div style="
            width: 64px; height: 64px; margin: 0 auto 12px;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, rgba(229,9,20,0.15), rgba(229,9,20,0.05));
            border: 1px solid rgba(229,9,20,0.35);
            border-radius: 16px; color: #E50914;">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/></svg>
        </div>
        <div style="color: #fafafa; font-weight: 700; font-size: 16px; letter-spacing: -0.3px;">Movie Intelligence</div>
        <div style="color: #8b8d97; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 4px;">Analyse & Recommandation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Classification", "Clusters", "Recommandations"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

df = load_features()
_nb_films = len(df) if df is not None else 0

st.sidebar.markdown(
    f"<div style='color: #8b8d97; font-size: 11px; "
    f"text-transform: uppercase; letter-spacing: 1.2px; padding: 0 4px;'>"
    f"Catalogue &nbsp; <span style='color:#E50914; font-weight: 700; font-size: 14px;'>"
    f"{_nb_films:,}</span> &nbsp; films</div>",
    unsafe_allow_html=True,
)

if df is None:
    st.error(
        "Fichier `data/features/movies_feature.csv` introuvable. "
        "Lance d'abord le notebook `04_feature_engineering.ipynb`."
    )
    st.stop()


if page == "Dashboard":
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{ICONS['sparkles']} TMDB Dataset</div>
            <h1>Movie Intelligence</h1>
            <p>Analyse intelligente du catalogue — Classification, Clustering & Recommandation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI Cards ---
    kpis = [
        kpi_card_html(
            "Films", f"{len(df):,}", "dans le catalogue",
            icon_key="film", trend="+12%", trend_type="success"
        ),
        kpi_card_html(
            "Note moyenne", f"{df['vote_average'].mean():.2f}", "sur 10",
            icon_key="star", trend="Top 30%", trend_type="gold"
        ),
        kpi_card_html(
            "Popularité médiane", f"{df['popularity'].median():.1f}", "score TMDB",
            icon_key="flame", trend="+8%", trend_type="success"
        ),
        kpi_card_html(
            "Durée médiane", f"{df['runtime'].median():.0f} min", "par film",
            icon_key="clock", trend="Stable", trend_type="info"
        ),
    ]
    render_kpis(kpis)

    # --- Charts Ligne 1 ---
    col1, col2 = st.columns(2)

    with col1:
        hist, edges = np.histogram(df["vote_average"].dropna(), bins=20, range=(0, 10))
        labels = [f"{edges[i]:.1f}" for i in range(len(edges) - 1)]
        render_chart(
            "chart_vote", "bar", labels, hist.tolist(),
            "Distribution des notes", color="#E50914",
        )

    with col2:
        hist, edges = np.histogram(
            df["popularity"].dropna(), bins=20,
            range=(0, df["popularity"].quantile(0.99)),
        )
        labels = [f"{edges[i]:.0f}" for i in range(len(edges) - 1)]
        render_chart(
            "chart_pop", "line", labels, hist.tolist(),
            "Distribution de la popularité", color="#3b82f6",
        )

    # --- Charts Ligne 2 ---
    col3, col4 = st.columns(2)

    with col3:
        exploded = df["genres"].explode().dropna()
        top_genres = exploded.value_counts().head(10)
        render_chart(
            "chart_genres", "bar",
            top_genres.index.tolist(), top_genres.values.tolist(),
            "Top 10 des genres", horizontal=True, multi_colors=True,
        )

    with col4:
        if "annee" in df.columns:
            years = df["annee"].dropna().astype(int)
            decades = (years // 10) * 10
            counts = decades.value_counts().sort_index()
            labels = [f"{int(d)}s" for d in counts.index]
            render_chart(
                "chart_decade", "bar", labels, counts.values.tolist(),
                "Films par décennie", color="#10b981",
            )

    # --- Filtres ---
    section_header("search", "Explorer le catalogue")

    c1, c2, c3 = st.columns(3)
    with c1:
        note_min = st.slider("Note minimum", 0.0, 10.0, 6.0, 0.1)
    with c2:
        pop_min = st.slider(
            "Popularité minimum",
            float(df["popularity"].min()),
            float(df["popularity"].max()),
            float(df["popularity"].median()),
        )
    with c3:
        langues = ["Toutes"] + sorted(df["original_language"].dropna().unique().tolist())
        langue = st.selectbox("Langue originale", langues)

    mask = (df["vote_average"] >= note_min) & (df["popularity"] >= pop_min)
    if langue != "Toutes":
        mask &= df["original_language"] == langue

    filtered = df[mask].sort_values("popularity", ascending=False)

    st.markdown(
        f'<div class="badge badge-success">{len(filtered)} films trouvés</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(
        filtered[["title", "release_date", "vote_average", "popularity", "runtime"]]
        .head(50).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


elif page == "Classification":
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{ICONS['target']} Machine Learning</div>
            <h1>Classification</h1>
            <p>Prédiction du <b>fort engagement</b> (vote_count ≥ médiane)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Entraîner les modèles"):
        with st.spinner("Entraînement en cours..."):
            from sklearn.model_selection import train_test_split
            from ml.classification import (
                create_target, prepare_data, create_preprocessor,
                create_models, train_models, evaluate_all_models,
            )

            df_target = create_target(df)
            X, y = prepare_data(df_target)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            preprocessor = create_preprocessor(X)
            models = create_models(preprocessor)
            models = train_models(models, X_train, y_train)
            all_results, cms = evaluate_all_models(models, X_test, y_test)

            st.session_state["clf_results"] = all_results

    if "clf_results" in st.session_state:
        results = st.session_state["clf_results"]

        section_header("chart", "Performances des modèles")

        kpis = []
        for name, metrics in results.items():
            kpis.append(kpi_card_html(
                name,
                f"{metrics['Accuracy']:.2%}",
                f"F1 : {metrics['F1-score']:.3f} • AUC : {metrics['ROC-AUC']:.3f}",
                icon_key="award",
                trend=f"AUC {metrics['ROC-AUC']:.2f}",
                trend_type="success",
            ))
        render_kpis(kpis)

        section_header("trending", "Comparaison détaillée")

        model_names = list(results.keys())
        accs = [results[m]["Accuracy"] for m in model_names]
        f1s = [results[m]["F1-score"] for m in model_names]
        aucs = [results[m]["ROC-AUC"] for m in model_names]

        chart_data = json.dumps({
            "labels": model_names,
            "datasets": [
                {"label": "Accuracy", "data": accs,
                 "backgroundColor": "#E50914", "borderRadius": 8},
                {"label": "F1", "data": f1s,
                 "backgroundColor": "#3b82f6", "borderRadius": 8},
                {"label": "ROC-AUC", "data": aucs,
                 "backgroundColor": "#10b981", "borderRadius": 8},
            ],
        })

        st.components.v1.html(
            f"""
            <div class="chart-card">
                <div class="chart-title">{ICONS['chart']}<span>Métriques comparées</span></div>
                <canvas id="clfChart" height="100"></canvas>
            </div>
            <script src="{CHARTJS_CDN}"></script>
            <script>
                new Chart(document.getElementById('clfChart'), {{
                    type: 'bar',
                    data: {chart_data},
                    options: {{
                        responsive: true,
                        animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                        plugins: {{
                            legend: {{ labels: {{ color: '#fafafa', font: {{ size: 12 }} }} }},
                            tooltip: {{
                                backgroundColor: 'rgba(20,22,30,0.98)',
                                titleColor: '#fafafa', bodyColor: '#fafafa',
                                borderColor: '#E50914', borderWidth: 1,
                                padding: 12, cornerRadius: 10,
                            }},
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#8b8d97' }},
                                  grid: {{ color: 'rgba(255,255,255,0.05)', drawBorder: false }} }},
                            y: {{ ticks: {{ color: '#8b8d97' }},
                                  grid: {{ color: 'rgba(255,255,255,0.05)', drawBorder: false }},
                                  beginAtZero: true, max: 1 }},
                        }},
                    }},
                }});
            </script>
            """,
            height=340,
        )
    else:
        st.info("Clique sur **Entraîner les modèles** pour lancer la classification.")


elif page == "Clusters":
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{ICONS['layers']} Unsupervised Learning</div>
            <h1>Clusters K-Means</h1>
            <p>Segmentation automatique des films par profil</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_clustered = load_clustered()

    if df_clustered is None:
        if st.button("Lancer le clustering"):
            with st.spinner("Clustering en cours..."):
                from ml.clustering import (
                    prepare_clustering_data, standardize_data,
                    test_k_values, select_best_k, apply_kmeans,
                )
                X = prepare_clustering_data(df)
                X_scaled, _ = standardize_data(X)
                results_df = test_k_values(X_scaled, k_values=range(2, 8))
                best_k = select_best_k(results_df)
                df_clustered, _ = apply_kmeans(X_scaled, df, best_k)
                df_clustered.to_csv(DATA_CLUSTERED, index=False)
    else:
        st.markdown(
            '<div class="badge badge-success">Données chargées depuis le cache</div>',
            unsafe_allow_html=True,
        )

    if df_clustered is not None and "cluster" in df_clustered.columns:
        counts = df_clustered["cluster"].value_counts().sort_index()

        kpis = [
            kpi_card_html("Clusters", f"{len(counts)}", "segments identifiés",
                          icon_key="layers", trend_type="info"),
            kpi_card_html("Films classifiés", f"{len(df_clustered):,}",
                          "dans le clustering",
                          icon_key="film", trend_type="success"),
            kpi_card_html("Cluster dominant", f"#{counts.idxmax()}",
                          f"{counts.max()} films",
                          icon_key="crown", trend="Top", trend_type="gold"),
        ]
        render_kpis(kpis)

        col1, col2 = st.columns(2)

        with col1:
            render_chart(
                "chart_cluster_count", "bar",
                [f"Cluster {i}" for i in counts.index],
                counts.values.tolist(),
                "Films par cluster",
                multi_colors=True,
            )

        with col2:
            clusters = sorted(df_clustered["cluster"].unique())
            palette = ["#E50914", "#3b82f6", "#10b981", "#f59e0b",
                       "#8b5cf6", "#ec4899", "#06b6d4"]
            datasets = []
            for i, c in enumerate(clusters):
                sub = df_clustered[df_clustered["cluster"] == c]
                datasets.append({
                    "label": f"Cluster {c}",
                    "data": [
                        {"x": float(r["popularity"]), "y": float(r["vote_average"])}
                        for _, r in sub.iterrows()
                    ],
                    "backgroundColor": palette[i % len(palette)],
                    "pointRadius": 4,
                    "pointHoverRadius": 8,
                })
            scatter_data = json.dumps({"datasets": datasets})

            st.components.v1.html(
                f"""
                <div class="chart-card">
                    <div class="chart-title">{ICONS['chart']}<span>Popularité vs Note</span></div>
                    <canvas id="scatterCluster" height="100"></canvas>
                </div>
                <script src="{CHARTJS_CDN}"></script>
                <script>
                    new Chart(document.getElementById('scatterCluster'), {{
                        type: 'scatter',
                        data: {scatter_data},
                        options: {{
                            responsive: true,
                            animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                            plugins: {{
                                legend: {{ labels: {{ color: '#fafafa', font: {{ size: 11 }} }} }},
                                tooltip: {{
                                    backgroundColor: 'rgba(20,22,30,0.98)',
                                    titleColor: '#fafafa', bodyColor: '#fafafa',
                                    borderColor: '#E50914', borderWidth: 1,
                                    padding: 12, cornerRadius: 10,
                                }},
                            }},
                            scales: {{
                                x: {{ title: {{ display: true, text: 'Popularité', color: '#8b8d97' }},
                                      ticks: {{ color: '#8b8d97' }},
                                      grid: {{ color: 'rgba(255,255,255,0.05)', drawBorder: false }} }},
                                y: {{ title: {{ display: true, text: 'Note', color: '#8b8d97' }},
                                      ticks: {{ color: '#8b8d97' }},
                                      grid: {{ color: 'rgba(255,255,255,0.05)', drawBorder: false }} }},
                            }},
                        }},
                    }});
                </script>
                """,
                height=340,
            )

        section_header("grid", "Profil moyen des clusters")
        num_cols = df_clustered.select_dtypes(include=[np.number]).columns
        cluster_means = df_clustered.groupby("cluster")[num_cols].mean().round(2)
        st.dataframe(cluster_means, use_container_width=True)

    else:
        st.info("Clique sur **Lancer le clustering** pour démarrer.")

elif page == "Recommandations":
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{ICONS['sparkles']} Content-Based</div>
            <h1>Recommandations</h1>
            <p>Système content-based par similarité cosinus (TF-IDF)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_reco, vectorizer, similarity_matrix = load_tfidf_reco()

    if df_reco is None:
        st.error("Impossible de charger les données de recommandation.")
        st.stop()

    col1, col2 = st.columns([3, 1])
    with col1:
        titles = sorted(df_reco["title"].dropna().unique().tolist())
        selected = st.selectbox("Choisis un film", titles)
    with col2:
        top_n = st.slider("Nombre", 3, 10, 5)

    if st.button("Recommander"):
        from ml.recommendation import recommend_movies

        with st.spinner("Calcul des similarités..."):
            recs = recommend_movies(selected, df_reco, similarity_matrix, top_n=top_n)

        if recs.empty:
            st.warning("Film introuvable dans le catalogue.")
        else:
            st.markdown(
                f'<div class="badge badge-success">'
                f'Top {len(recs)} films similaires à <b>{selected}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            for i, row in recs.iterrows():
                st.markdown(
                    movie_card_html(
                        rank=i + 1,
                        title=row["title"],
                        genres=row["genres"] if isinstance(row["genres"], list) else [],
                        year=int(row["annee"]) if pd.notna(row["annee"]) else "N/A",
                        score=row["similarity_score"],
                    ),
                    unsafe_allow_html=True,
                )