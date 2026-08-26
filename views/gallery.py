from __future__ import annotations

import streamlit as st

import config
from src.catalog import group_by_aoi, scan, thumbnail_bytes
from views.common import hero, render_footer, render_shell, safe_page_link

render_shell()
hero(
    '<div class="satquery-title">Dataset <span>Gallery</span></div>'
    "<div class=\"satquery-sub\">Co-registered Sentinel-style scenes &mdash; browse, inspect, and load "
    "into the analysis session</div>"
)

col_refresh, col_info = st.columns([1, 3])
if col_refresh.button("🔄 Rescan dataset folder", width="stretch"):
    st.cache_data.clear()
    st.rerun()

uploads = st.file_uploader(
    "Add your own GeoTIFFs (optical 4-band blue/green/red/nir · SAR vv/vh)",
    type=["tif", "tiff"], accept_multiple_files=True,
)
if uploads:
    for up in uploads:
        (config.RAW_DIR / f"upload_{up.name}").write_bytes(up.getvalue())
    st.cache_data.clear()
    st.toast(f"Imported {len(uploads)} file(s)", icon="✅")
    st.rerun()

entries = scan()
grouped = group_by_aoi(entries)

if not grouped:
    st.info("No scenes found in data/raw. Run `uv run python scripts/make_synthetic.py`.")
    st.stop()

st.caption(
    f"{len(entries)} scenes · {len(grouped)} AOIs · synthetic S1/S2 pairs over Indian cities · "
    "EPSG:4326 georeferenced · real Sentinel data can be added via scripts/fetch_data.py"
)

for aoi, slots in grouped.items():
    with st.expander(f"📍 {aoi.title()} — {slots.get('_source', 'synthetic')}", expanded=False):
        cols = st.columns(3)
        slot_meta = [
            ("t1", "Optical T1", "🟦"),
            ("t2", "Optical T2", "🟩"),
            ("sar", "SAR VV/VH", "🟥"),
        ]
        for col, (slot, label, dot) in zip(cols, slot_meta):
            entry = slots.get(slot)
            with col:
                if entry is None:
                    st.markdown(f"**{label}**")
                    st.caption("not present")
                    continue
                try:
                    st.image(thumbnail_bytes(entry), width="stretch")
                except Exception:
                    st.warning("thumbnail unavailable")
                st.markdown(f"**{dot} {label}** — `{entry.scene_id}`")
                st.caption(
                    f"{entry.kind.upper()} · bands: {', '.join(entry.bands) or '?'} · "
                    f"{entry.shape[1]}×{entry.shape[2]} px · {entry.crs}\n\n"
                    f"bbox {entry.bounds}"
                )
                session_key = {"t1": "optical_t1_path", "t2": "optical_t2_path", "sar": "sar_path"}[slot]
                current = st.session_state.get(session_key)
                is_loaded = current == entry.path
                if is_loaded:
                    st.success("Loaded ✓")
                elif st.button("Load into session", key=f"load_{aoi}_{slot}", width="stretch"):
                    st.session_state[session_key] = entry.path
                    st.toast(f"{aoi.title()} {label} loaded", icon="🛰️")
                    st.rerun()

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Optical T1 loaded", "✔" if st.session_state.get("optical_t1_path") else "—")
c2.metric("Optical T2 loaded", "✔" if st.session_state.get("optical_t2_path") else "—")
c3.metric("SAR loaded", "✔" if st.session_state.get("sar_path") else "—")
safe_page_link("views/chat.py", label="💬 Go to Chat Analysis →", icon="➡️")
safe_page_link("views/studio3d.py", label="🌐 Open 3D Studio →", icon="➡️")
render_footer()
