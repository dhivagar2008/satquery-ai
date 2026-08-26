from __future__ import annotations

import streamlit as st

import config
from views.common import hero, render_footer, render_shell

render_shell()
hero(
    '<div class="satquery-title">About <span>SatQuery AI</span></div>'
    "<div class=\"satquery-sub\">Architecture, stack, and design decisions &mdash; SIH26167</div>"
)

st.subheader("Problem statement")
st.markdown(
    "> **SIH26167 — ISRO:** Build an interactive vision-language assistant that lets analysts query "
    "multimodal remote-sensing imagery (Optical + SAR) through natural language, with automated "
    "routing to specialist analysis engines and explainable outputs."
)

left, right = st.columns(2)
with left:
    st.subheader("Tech stack")
    st.markdown(
        "| Layer | Technology |\n|---|---|\n"
        "| Frontend | Streamlit (multi-page, dark) |\n"
        "| Backend | FastAPI + Uvicorn (`/api/query`) |\n"
        "| GIS engine | rasterio · OpenCV · scikit-image · shapely |\n"
        "| Agentic router | Groq JSON-mode LLM + heuristic fallback |\n"
        "| VQA | Groq Vision (Llama-4 family) → rule-based CV fallback |\n"
        "| Auth | Google OAuth 2.0 (streamlit-oauth) + guest mode |\n"
        "| 3D visualization | pydeck (deck.gl Terrain/Bitmap/Column layers) · Plotly surfaces |\n"
        "| Data | Synthetic S1/S2 generator · Planetary Computer STAC fetcher |"
    )
with right:
    st.subheader("Design principles")
    st.markdown(
        "- **Never fail live:** every LLM call degrades to deterministic CV/rule logic\n"
        "- **CPU-first:** heavy inference is API-side; local pipeline is NumPy/OpenCV only\n"
        "- **Explainable:** every answer ships the router decision trace JSON\n"
        "- **Radiometric honesty:** bi-temporal comparisons use shared normalization so unchanged "
        "pixels stay identical\n- **Georeferenced metrics:** areas in km² derived from CRS-aware "
        "pixel geometry, not raw pixel counts"
    )

st.subheader("The four specialist engines")
e1, e2, e3, e4 = st.columns(4)
for col, title, body in [
    (e1, "🔍 VQA Single", "Vision-language model describes scenes; falls back to edge-density + "
     "spectral-proxy statistics rendered as a structured report."),
    (e2, "🔀 Cross-Modal Fusion", "NDWI water candidates cross-checked against dark SAR backscatter; "
     "NDBI-style built-up confirmed by bright VH texture. Reports SAR-rejected false positives."),
    (e3, "⏱️ Change Detection", "Shared radiometric normalization → abs-diff ⊕ SSIM dissimilarity "
     "(noise-gated) → Otsu mask → morphology → km² change stats."),
    (e4, "🎯 Segmentation", "Index-threshold masks (NDWI>0.08 water, NDVI>0.34 vegetation, "
     "brightness-texture built-up) cleaned by morphological open/close + component filtering."),
]:
    col.markdown(f"**{title}**\n\n{body}")

st.subheader("Dataset")
st.markdown(
    "Bundled: **15 georeferenced GeoTIFFs** — synthetic co-registered Sentinel-style pairs "
    "(optical B/G/R/NIR + SAR VV/VH, uint16, EPSG:4326, 512×512 px ≈ 0.2°) over "
    "**Chennai, Bengaluru, Mumbai, Delhi, Kolkata**, each with a realistic urban-expansion scenario. "
    "Real Sentinel-1/2 can be pulled from Microsoft Planetary Computer via "
    "`scripts/fetch_data.py` on an open network."
)

st.caption(f"LLM status: {'🟢 ' + config.GROQ_TEXT_MODEL if config.llm_available() else '🟡 offline mode — set GROQ_API_KEY in .env'}")
render_footer()
