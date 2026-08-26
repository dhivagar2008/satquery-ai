from __future__ import annotations

import streamlit as st

import config
from src.catalog import group_by_aoi, scan
from views.common import hero, render_footer, render_shell

render_shell()
hero(
    '<div class="satquery-title">SatQuery <span>AI</span> &mdash; Mission Control</div>'
    "<div class=\"satquery-sub\">Interactive Vision-Language Assistant for Multimodal Remote Sensing "
    "Image Analysis through Text Queries &mdash; built for ISRO (Smart India Hackathon SIH26167)</div>"
)

t1_loaded = bool(st.session_state.get("optical_t1_path"))
sar_loaded = bool(st.session_state.get("sar_path"))
llm_on = config.llm_available()

entries = scan()
grouped = group_by_aoi(entries)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenes in Dataset", len(entries))
c2.metric("Cities (AOIs)", len(grouped))
c3.metric("CV Engines", "4 online")
c4.metric("Groq LLM", "🟢 Online" if llm_on else "🟡 Rule fallback")

st.markdown("")
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(
        '<div class="feature-card"><h4>💬 Chat Analysis</h4>'
        "<p>Ask natural-language questions about any loaded scene. An agentic LLM router picks the "
        "right specialist engine and returns answers with metrics and overlay PNGs.</p></div>",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        '<div class="feature-card"><h4>🛰️ Dataset Gallery</h4>'
        f"<p>{len(entries)} georeferenced Sentinel-style scenes across {len(grouped)} Indian cities "
        "(synthetic S1/S2 pairs). Browse thumbnails, inspect metadata, load scenes into session.</p></div>",
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        '<div class="feature-card"><h4>🌐 3D Studio</h4>'
        "<p>Photo-drape imagery over 3D terrain, extrude classified land-cover into columns, fly "
        "NDVI/NDWI surfaces, and explore bi-temporal change as a height field.</p></div>",
        unsafe_allow_html=True,
    )
with f4:
    st.markdown(
        '<div class="feature-card"><h4>🔌 REST API</h4>'
        "<p>FastAPI backend at <code>:8001/api/query</code> exposes the full pipeline — multipart "
        "uploads in, JSON + base64 overlays out. Swagger docs included.</p></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("How it works")
st.markdown(
    """```
User Query + GeoTIFFs ──▶ Streamlit UI / FastAPI (:8501 · :8001)
     │
     ▼
Agentic Router ── Groq JSON mode ──┐
     │  (heuristic fallback)       │ intent: VQA | FUSION | CHANGE | SEGMENT
     ▼                             ▼
 ┌────────────── Specialist CV Engines ──────────────┐
 │ VQA (vision LLM)   Cross-Modal Fusion (SAR×Optical)│
 │ Change Detection   Spatial Segmentation            │
 └───────────────────────┬───────────────────────────┘
                         ▼
 GIS Pipeline (rasterio · NDVI/NDWI · Otsu · SSIM · morphology)
                         ▼
 Answer Synthesizer ──▶ Markdown answer + spatial metrics + PNG overlays
```"""
)

st.info(
    "**Quick start:** open the 🛰️ Dataset Gallery → load a city pair → ask "
    "*“Highlight water bodies”* in 💬 Chat Analysis → view results in 🌐 3D Studio.",
    icon="⚡",
)
render_footer()
