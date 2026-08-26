from __future__ import annotations

import base64
import json

import numpy as np
import streamlit as st

import config
from src.gis.pipeline import load_geotiff, normalize_band, to_rgb_render
from src.orchestrator.router import route_query
from src.orchestrator.schemas import QueryContext
from src.tools.registry import execute_pipeline
from views.common import INTENT_BADGES, hero, render_footer, render_shell, safe_page_link

render_shell()
hero(
    '<div class="satquery-title">Chat <span>Analysis</span></div>'
    "<div class=\"satquery-sub\">Natural-language queries → agentic routing → specialist CV engines "
    "→ answers with metrics &amp; overlays</div>"
)

RAW_T1 = config.RAW_DIR / "chennai_optical_t1.tif"
RAW_T2 = config.RAW_DIR / "chennai_optical_t2.tif"
RAW_SAR = config.RAW_DIR / "chennai_sar.tif"

status = {
    "t1": st.session_state.get("optical_t1_path"),
    "t2": st.session_state.get("optical_t2_path"),
    "sar": st.session_state.get("sar_path"),
}
loaded_any = any([status["t1"], status["t2"], status["sar"]])

EXAMPLE_QUERIES = [
    "Describe the land cover in this image",
    "Highlight water bodies",
    "What changed between T1 and T2?",
    "Has urban area expanded between the two dates?",
    "Use the optical and SAR images together to identify built-up and water-covered regions",
    "Mask all agricultural zones",
]

if not loaded_any:
    st.warning("No scene loaded — load a city pair to enable all four engines.", icon="⚠️")
    if RAW_T1.exists() and st.button("⚡ Load Chennai demo pair now", type="primary"):
        st.session_state.optical_t1_path = str(RAW_T1)
        st.session_state.optical_t2_path = str(RAW_T2)
        st.session_state.sar_path = str(RAW_SAR)
        st.toast("Chennai pair loaded", icon="🛰️")
        st.rerun()
    safe_page_link("views/gallery.py", label="Browse the Dataset Gallery →", icon="🛰️")
    st.stop()

try:
    optical_raster = load_geotiff(status["t1"]) if status["t1"] else None
    sar_raster = load_geotiff(status["sar"]) if status["sar"] else None
    t2_raster = load_geotiff(status["t2"]) if status["t2"] else None

    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.markdown("**Optical T1 (S2 RGB)**")
        if optical_raster is not None:
            st.image(to_rgb_render(optical_raster), width="stretch")
        else:
            st.caption("—")
    with preview_cols[1]:
        st.markdown("**SAR (S1 VV/VH)**")
        if sar_raster is not None:
            vv = normalize_band(sar_raster.band("vv"))
            vh = normalize_band(sar_raster.band("vh"))
            st.image(np.dstack([vv, vh, np.full_like(vv, 128)]), width="stretch")
        else:
            st.caption("not loaded — fusion unavailable")
    with preview_cols[2]:
        st.markdown("**Optical T2**")
        if t2_raster is not None:
            st.image(to_rgb_render(t2_raster), width="stretch")
        else:
            st.caption("not loaded — change detection unavailable")

    badges = f'<span class="badge">LLM: {"Groq online" if config.llm_available() else "offline rule engines"}</span>'
    if sar_raster is not None:
        badges += '<span class="badge">Fusion ready</span>'
    if t2_raster is not None:
        badges += '<span class="badge">Bi-temporal ready</span>'
    st.markdown(badges, unsafe_allow_html=True)
    st.divider()
except ValueError as exc:
    st.error(f"Raster load error: {exc}")
    st.stop()

for msg in st.session_state.messages:
    avatar = "🧑‍🔬" if msg["role"] == "user" else "🛰️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("intent"):
            badge_label = INTENT_BADGES.get(msg["intent"], (msg["intent"], ""))[0]
            st.markdown(
                f'<span class="badge">{badge_label}</span>'
                f'<span class="badge">Router: {msg.get("router_source", "")}</span>'
                f'<span class="badge">Engine: {msg.get("engine", "")}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            mcols = st.columns(max(len(msg.get("metrics") or []), 1))
            for col, metric in zip(mcols, msg.get("metrics") or []):
                col.metric(metric["label"], metric["value"])
            if msg.get("overlays"):
                img_cols = st.columns(min(len(msg["overlays"]), 3))
                for col, (name, data_url) in zip(img_cols, msg["overlays"].items()):
                    col.markdown(f"`{name}`")
                    col.image(data_url, width="stretch")
                    col.download_button(
                        "⬇ Download PNG",
                        base64.b64decode(data_url.split(",", 1)[-1]),
                        file_name=f"{name}.png", mime="image/png",
                        key=f"dl_{name}_{id(msg)}", width="stretch",
                    )
            if msg.get("trace"):
                with st.expander("🧠 Agentic Router Decision Trace (JSON)"):
                    st.json(msg["trace"])

user_input = st.chat_input(
    "Ask about the imagery… e.g. 'Highlight water bodies' or 'What changed between T1 and T2?'"
)

chip_cols = st.columns(len(EXAMPLE_QUERIES))
for col, q in zip(chip_cols, EXAMPLE_QUERIES):
    if col.button(q, key=f"chip_{q}", width="stretch"):
        user_input = q

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(user_input)

    ctx = QueryContext(
        user_query=user_input,
        has_optical=status["t1"] is not None,
        has_sar=status["sar"] is not None,
        is_bitemporal=status["t2"] is not None,
    )
    with st.chat_message("assistant", avatar="🛰️"):
        with st.spinner("Routing intent → executing specialist engine → synthesizing answer…"):
            try:
                intent_result = route_query(ctx)
                output = execute_pipeline(
                    intent_result,
                    optical=load_geotiff(status["t1"]) if status["t1"] else None,
                    sar=load_geotiff(status["sar"]) if status["sar"] else None,
                    optical_t2=load_geotiff(status["t2"]) if status["t2"] else None,
                    user_query=user_input,
                )
                report_md = (
                    f"# SatQuery AI Report\n\n**Query:** {user_input}\n\n"
                    f"- Intent: `{intent_result.primary_intent}` (router: {intent_result.source})\n"
                    f"- Engine: `{output.engine_name}`\n\n## Metrics\n"
                    + "\n".join(f"- **{m.label}:** {m.value}" for m in output.metrics)
                    + f"\n\n## Answer\n\n{output.answer_markdown}\n\n"
                    + "## Router Trace\n```json\n"
                    + json.dumps(intent_result.model_dump(), indent=2)
                    + "\n```\n"
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": output.answer_markdown,
                    "intent": intent_result.primary_intent,
                    "router_source": intent_result.source,
                    "engine": output.engine_name,
                    "metrics": [m.model_dump() for m in output.metrics],
                    "overlays": output.overlays,
                    "trace": intent_result.model_dump(),
                    "report": report_md,
                }
                st.session_state.messages.append(assistant_msg)
                st.rerun()
            except ValueError as exc:
                err = f"**Execution error:** {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

render_footer()
