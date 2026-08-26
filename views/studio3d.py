from __future__ import annotations

import socket

import streamlit as st

import config
from src.catalog import scan
from src.engines import fusion_engine, segmentation
from src.gis.pipeline import load_geotiff
from src.viz3d import (
    build_class_columns_deck,
    build_photo_drape_deck,
    plotly_change_surface,
    plotly_index_surface,
)
from views.common import hero, render_footer, render_shell

render_shell()
hero(
    '<div class="satquery-title">3D <span>Studio</span></div>'
    "<div class=\"satquery-sub\">Photo-draped terrain · land-cover column extrusions · index surfaces · "
    "bi-temporal change fields</div>"
)


@st.cache_data(ttl=600, show_spinner=False)
def internet_available() -> bool:
    try:
        s = socket.create_connection(("s3.amazonaws.com", 443), timeout=2.5)
        s.close()
        return True
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def cached_segment(path_str: str, feature: str):
    raster = load_geotiff(path_str)
    return segmentation.segment_feature(raster, feature)["mask"]


@st.cache_data(show_spinner=False)
def cached_fusion(opt_path: str, sar_path: str):
    return fusion_engine.fuse_analysis(load_geotiff(opt_path), load_geotiff(sar_path))


@st.cache_data(show_spinner=False)
def cached_change_fig(t1_path: str, t2_path: str, step: int, exaggeration: float):
    fig = plotly_change_surface(load_geotiff(t1_path), load_geotiff(t2_path),
                                step=step, exaggeration=exaggeration)
    return fig.to_json()


@st.cache_data(show_spinner=False)
def cached_index_fig(path: str, index: str, step: int, exaggeration: float):
    fig = plotly_index_surface(load_geotiff(path), index, step=step, exaggeration=exaggeration)
    return fig.to_json()


def show_deck(builder):
    deck = builder()
    spec = deck.to_json()
    st.caption(f"deck payload: {len(spec) / 1024:.0f} KB")
    if len(spec) > 6_000_000:
        st.error("Deck payload too large for the browser — reduce resolution or sample size.")
        return
    st.pydeck_chart(deck, width="stretch")


entries = {e.scene_id: e for e in scan() if e.kind == "optical"}
if not entries:
    st.info("No optical scenes found — run `uv run python scripts/make_synthetic.py` first.")
    st.stop()

optical_options = sorted([sid for sid, e in entries.items() if e.slot in ("t1", "t2")])
sar_by_aoi = {e.aoi: e for e in scan() if e.kind == "sar"}

scene_id = st.selectbox("Scene (Optical)", optical_options,
                        format_func=lambda s: s.replace("_", " ").title())
entry = entries[scene_id]
aoi = entry.aoi
sar_entry = sar_by_aoi.get(aoi)
online = internet_available()
if not online:
    st.warning("No internet detected — DEM terrain tiles unavailable; "
               "photo-drape will use flat mode and surfaces/extrusions work fully offline.", icon="📡")

tab_drape, tab_surface, tab_columns, tab_change = st.tabs(
    ["🛰️ Photo-Drape Terrain", "📈 Index Surface", "🏗️ Class Extrusion", "⏱️ Change 3D"]
)

raster = load_geotiff(entry.path)

with tab_drape:
    try:
        d1, d2, d3 = st.columns([2, 1, 1])
        overlay_choice = d1.select_slider(
            "Overlay layer",
            options=["None", "Water", "Vegetation", "Built-up", "Clouds"],
            value="None",
        )
        use_terrain = d2.toggle("DEM relief", value=False,
                                help="Experimental: adds real elevation tiles from AWS "
                                     "(internet required). Turn off if the map fails to render.")
        pitch = d3.slider("Camera pitch", 0, 80, 55)

        mask_map = {
            "Water": ("water", (30, 90, 255)),
            "Vegetation": ("vegetation", (40, 190, 70)),
            "Built-up": ("built_up", (255, 120, 30)),
            "Clouds": ("cloud", (240, 240, 240)),
        }
        overlay_mask = None
        mask_color = (255, 40, 40)
        if overlay_choice != "None":
            feature, mask_color = mask_map[overlay_choice]
            with st.spinner(f"Segmenting {feature}…"):
                overlay_mask = cached_segment(entry.path, feature)

        with st.spinner("Building photo-drape…"):
            show_deck(lambda: build_photo_drape_deck(
                raster, scene_key=f"{scene_id}_{overlay_choice}_{use_terrain}",
                overlay_mask=overlay_mask, mask_color=mask_color,
                use_terrain=use_terrain, pitch=pitch))
        st.caption("Drag to orbit · scroll to zoom · right-drag to tilt. Texture served "
                   "from the app's local static server — works fully offline.")
    except Exception as exc:
        st.error(f"Photo-drape failed: {exc}")

with tab_surface:
    try:
        names = [n.lower() for n in raster.band_names]
        available = ["brightness"]
        if {"red", "nir"} <= set(names):
            available.append("ndvi")
        if {"green", "nir"} <= set(names):
            available.append("ndwi")
        c1, c2 = st.columns(2)
        idx = c1.select_slider("Index / band", options=available,
                               value="ndvi" if "ndvi" in available else "brightness")
        exaggeration = c2.slider("Height exaggeration", 5, 60, 25, key="surf_ex")
        import plotly.io as pio

        fig_json = cached_index_fig(entry.path, idx, 4, float(exaggeration))
        st.plotly_chart(pio.from_json(fig_json), width="stretch")
    except Exception as exc:
        st.error(f"Index surface failed: {exc}")

with tab_columns:
    try:
        if sar_entry is None:
            st.warning(f"No SAR scene for AOI '{aoi}' — showing optical-only class proxies.")
            masks = {feat: cached_segment(entry.path, feat)
                     for feat in ("water", "vegetation", "built_up")}
        else:
            with st.spinner("Running cross-modal fusion…"):
                masks = cached_fusion(entry.path, sar_entry.path)["masks"]
        class_heights = {"water": 40.0, "vegetation": 120.0, "built_up": 260.0}
        ex = st.slider("Extrusion exaggeration ×", 1, 15, 6, key="col_ex")
        show_deck(lambda: build_class_columns_deck(
            raster, masks, class_heights, exaggeration=float(ex)))
        st.markdown(
            '<span class="badge" style="background:#1e3a8a">🔵 Water</span>'
            '<span class="badge" style="background:#14532d">🟢 Vegetation</span>'
            '<span class="badge" style="background:#7c2d12">🟠 Built-up</span>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Column extrusion failed: {exc}")

with tab_change:
    try:
        t2_candidates = [e for e in scan() if e.kind == "optical" and e.slot == "t2" and e.aoi == aoi]
        if entry.slot != "t1" or not t2_candidates:
            st.warning("Change 3D requires this scene to be an Optical **T1** with a matching T2 pair.")
        else:
            t2_entry = t2_candidates[0]
            c_ex = st.slider("Change height scale", 10, 100, 40, key="chg_ex")
            import plotly.io as pio

            fig_json = cached_change_fig(entry.path, t2_entry.path, 4, float(c_ex))
            st.plotly_chart(pio.from_json(fig_json), width="stretch")
            st.caption("Red peaks = structural change between T1 → T2 (diff ⊕ SSIM, noise-gated).")
    except Exception as exc:
        st.error(f"Change surface failed: {exc}")

st.divider()
st.caption(
    "Textures are served from the app's local static folder (offline-safe). DEM terrain tiles from "
    "AWS require internet; all other 3D modes work fully offline."
)
render_footer()
