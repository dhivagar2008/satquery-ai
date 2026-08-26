from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from . import lite_engine as le
except ImportError:
    import lite_engine as le

app = FastAPI(
    title="SatQuery AI — Lite Serverless API",
    description="Public demo engine for multimodal satellite imagery queries (ISRO / SIH26167)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "lite-serverless",
        "intents": ["VQA_SINGLE", "SPATIAL_SEGMENTATION", "CHANGE_DETECTION", "CROSS_MODAL_FUSION"],
        "author": "Dhivagar R",
    }


@app.post("/api/query")
async def query(
    query: str = Form(...),
    optical_file: UploadFile | None = File(None),
    sar_file: UploadFile | None = File(None),
    bitemporal_file: UploadFile | None = File(None),
):
    if optical_file is None or not optical_file.filename:
        raise HTTPException(422, "optical_file is required")

    opt_bytes = await optical_file.read()
    sar_bytes = await sar_file.read() if (sar_file and sar_file.filename) else None
    t2_bytes = await bitemporal_file.read() if (bitemporal_file and bitemporal_file.filename) else None

    try:
        opt = le.read_tiff(opt_bytes)
    except Exception as exc:
        raise HTTPException(422, f"Could not parse optical GeoTIFF: {exc}") from exc
    sar = le.read_tiff(sar_bytes) if sar_bytes else None
    t2 = le.read_tiff(t2_bytes) if t2_bytes else None

    intent, reasoning = le.route_intent(query, True, sar is not None, t2 is not None)

    overlays: dict[str, str] = {}
    metrics: list[dict] = []
    parts: dict = {"scene": opt}

    if intent == "SPATIAL_SEGMENTATION":
        feature = le.normalize_feature(le.extract_target(query)) or "water"
        mask, canonical = le.segment_mask(opt, feature)
        if mask is None:
            raise HTTPException(422, f"Cannot segment '{feature}' — required bands missing.")
        km2 = le.pixel_km2(opt)
        stats = le.mask_stats(mask, km2)
        parts.update({"stats": stats, "feature": canonical})
        overlays[f"segmentation_{canonical}_overlay"] = le.tint(
            le.rgb_render(opt), mask, {"water": (30, 90, 255),
                                       "vegetation": (40, 190, 70),
                                       "built_up": (255, 120, 30),
                                       "cloud": (240, 240, 240)}.get(canonical, (255, 40, 40)))
        metrics = [
            {"label": "Feature Class", "value": canonical.replace("_", " ").title()},
            {"label": "Coverage", "value": f"{stats['percent']}%"},
            {"label": "Area", "value": f"{stats.get('area_km2', 'N/A')} km²" if km2 else "px only"},
            {"label": "Engine", "value": "Lite CV"},
        ]

    elif intent == "CHANGE_DETECTION":
        if t2 is None:
            raise HTTPException(422, "Change detection needs bitemporal_file (T2 image).")
        res = le.change_analysis(opt, t2)
        parts["stats"] = res["stats"]
        overlays["change_overlay_t2"] = res["overlay_b64"]
        arrow = {"increased": "▲", "decreased": "▼", "stable": "■"}[res["stats"]["net_direction"]]
        metrics = [
            {"label": "Change", "value": f"{res['stats']['change_percent']}%"},
            {"label": "Net Signal", "value": f"{arrow} {res['stats']['net_direction'].title()}"},
            {"label": "Threshold", "value": str(res["stats"]["threshold"])},
            {"label": "Engine", "value": "Lite CV"},
        ]
        if "changed_area_km2" in res["stats"]:
            metrics.insert(1, {"label": "Changed Area",
                               "value": f"{res['stats']['changed_area_km2']} km²"})

    elif intent == "CROSS_MODAL_FUSION":
        if sar is None:
            raise HTTPException(422, "Fusion needs sar_file (S1 VV/VH GeoTIFF).")
        res = le.fusion_analysis(opt, sar)
        parts["stats"] = res["stats"]
        overlays["fusion_landcover_overlay"] = res["overlay_b64"]
        f = res["stats"]
        cm = f.get("cross_modal", {})
        row = lambda k: (f"{f[k]['percent']}%" + (f" ({f[k]['area_km2']} km²)"
                         if "area_km2" in f[k] else "")) if k in f else "—"
        metrics = [
            {"label": "Water", "value": row("water")},
            {"label": "Built-up", "value": row("built_up")},
            {"label": "Vegetation", "value": row("vegetation")},
            {"label": "SAR Rejections", "value": f"{cm.get('sar_rejection_percent', 0)}% FP"},
        ]

    else:
        overlays["optical_render"] = le.tint(le.rgb_render(opt), None, (0, 0, 0))
        metrics = [
            {"label": "Answer Source", "value": "Rule-based Lite"},
            {"label": "Water %", "value": f"{(le.ndwi(opt) > 0.05).mean() * 100:.2f}"
             if le.ndwi(opt) is not None else "—"},
            {"label": "Vegetation %", "value": f"{(le.ndvi(opt) > 0.38).mean() * 100:.2f}"
             if le.ndvi(opt) is not None else "—"},
        ]

    answer = le.markdown_answer(intent, query, parts)

    return {
        "intent": intent,
        "reasoning": reasoning,
        "router_source": "heuristic-lite",
        "tool_pipeline": [{"step": 1, "tool_name": f"lite_{intent.lower()}", "parameters": {}}],
        "requires_change_map": intent == "CHANGE_DETECTION",
        "engine": "lite-serverless",
        "answer_markdown": answer,
        "metrics": metrics,
        "overlays_b64": overlays,
        "stats": {k: v for k, v in parts.items() if k != "scene"} | {"scene_shape": opt["shape"]},
    }
