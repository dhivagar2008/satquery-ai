from __future__ import annotations

import streamlit as st

import config
from src.gis.pipeline import load_geotiff
from views.auth import enforce_login

PAGE_CSS = """
<style>
    .block-container { padding-top: 1.6rem; max-width: 1500px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg,#111b2e,#0d1526);
        border: 1px solid #1e2a45; border-radius: 12px; padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] p { color:#7dd3fc !important; font-size:0.82rem; }
    div[data-testid="stExpander"] { border-color:#1e2a45; }
    .satquery-header {
        background: radial-gradient(ellipse at top left, rgba(6,182,212,.18), transparent 60%),
                    linear-gradient(135deg,#0b1120 0%,#111b2e 100%);
        border: 1px solid #1e2a45; border-radius: 14px;
        padding: 1.3rem 2rem; margin-bottom: 1rem;
    }
    .satquery-title { font-size: 2.1rem; font-weight: 800; color:#f8fafc; letter-spacing:-.5px;}
    .satquery-title span { color:#06b6d4; }
    .satquery-sub { color:#94a3b8; font-size:.95rem; margin-top:.25rem;}
    .badge {
        display:inline-block; padding:.18rem .7rem; border-radius:999px;
        font-size:.78rem; font-weight:600; margin-right:.4rem;
        border:1px solid #334155; color:#e2e8f0; background:#16223a;
    }
    .feature-card {
        background:linear-gradient(145deg,#101a2d,#0c1424);
        border:1px solid #1e2a45; border-radius:14px; padding:1.2rem 1.4rem; height:100%;
    }
    .feature-card h4 { color:#7dd3fc; margin:0 0 .5rem 0; font-size:1.05rem;}
    .feature-card p { color:#94a3b8; font-size:.86rem; margin:0;}
</style>
"""


def render_shell():
    if not st.session_state.get("_shell_ready"):
        st.set_page_config(
            page_title="SatQuery AI | ISRO Remote Sensing Assistant",
            page_icon="\U0001F6F0️",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_shell_ready"] = True
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    defaults = {
        "messages": [],
        "optical_t1_path": None,
        "optical_t2_path": None,
        "sar_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    enforce_login()


def render_footer():
    st.markdown(
        f"""<div style="text-align:center; color:#475569; font-size:.78rem;
             padding:1.2rem 0 .4rem; border-top:1px solid #16223a; margin-top:2.2rem;">
             🛰️ SatQuery AI &nbsp;·&nbsp; Interactive Vision-Language Assistant for ISRO Remote Sensing
             &nbsp;·&nbsp; SIH26167<br>
             Published by <b style="color:#7dd3fc">{config.APP_AUTHOR}</b></div>""",
        unsafe_allow_html=True,
    )


def hero(title_html: str | None = None):
    if title_html is None:
        title_html = (
            '<div class="satquery-title">SatQuery <span>AI</span></div>'
            '<div class="satquery-sub">Interactive Vision-Language Assistant for Multimodal Remote '
            "Sensing Image Analysis &mdash; built for ISRO (SIH26167)</div>"
        )
    st.markdown(f'<div class="satquery-header">{title_html}</div>', unsafe_allow_html=True)


def load_session_rasters() -> dict:
    out = {"optical": None, "optical_t2": None, "sar": None, "errors": []}
    for key, attr in [("optical", "optical_t1_path"), ("optical_t2", "optical_t2_path"),
                      ("sar", "sar_path")]:
        path = st.session_state.get(attr)
        if not path:
            continue
        try:
            out[key] = load_geotiff(path)
        except ValueError as exc:
            out["errors"].append(f"{attr}: {exc}")
    return out


def session_status() -> dict:
    return {
        "t1": st.session_state.get("optical_t1_path"),
        "t2": st.session_state.get("optical_t2_path"),
        "sar": st.session_state.get("sar_path"),
        "llm": config.llm_available(),
    }


def safe_page_link(page_path: str, label: str, icon: str | None = None):
    try:
        st.page_link(page_path, label=label, icon=icon)
    except Exception:
        st.caption(f"➡ {label}")


INTENT_BADGES = {
    "VQA_SINGLE": ("\U0001F50D VQA Single Image", "cyan"),
    "CROSS_MODAL_FUSION": ("\U0001F500 Cross-Modal Fusion", "violet"),
    "CHANGE_DETECTION": ("\u23F1️ Change Detection", "orange"),
    "SPATIAL_SEGMENTATION": ("\U0001F3AF Spatial Segmentation", "green"),
}
