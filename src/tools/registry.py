from __future__ import annotations

import base64

from src.engines import change_detection, fusion_engine, segmentation, vqa_engine
from src.gis.pipeline import RasterData
from src.orchestrator.schemas import EngineOutput, IntentResult, MetricValue
from src.synthesis.synthesizer import synthesize


def _png_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def execute_pipeline(intent_result: IntentResult,
                     optical: RasterData | None = None,
                     sar: RasterData | None = None,
                     optical_t2: RasterData | None = None,
                     user_query: str = "") -> EngineOutput:
    intent = intent_result.primary_intent
    stats: dict = {}
    overlays: dict[str, bytes] = {}
    metrics: list[MetricValue] = []
    engine_name = "unknown"

    if intent == "SPATIAL_SEGMENTATION":
        if optical is None:
            raise ValueError("Segmentation requires an optical (Sentinel-2) image.")
        params = next((s.parameters for s in intent_result.tool_pipeline
                       if s.tool_name == "tool_spatial_segmentation"), {})
        result = segmentation.segment_feature(optical, params.get("target_feature"))
        engine_name = "spatial_segmentation"
        stats["feature"] = result["feature"]
        stats["segmentation"] = result["stats"]
        overlays[f"segmentation_{result['feature']}_overlay"] = result["overlay_png"]
        s = result["stats"]
        metrics = [
            MetricValue(label="Feature Class", value=result["feature"].replace("_", " ").title()),
            MetricValue(label="Area", value=f"{s['area_km2']} km²"),
            MetricValue(label="Coverage", value=f"{s['percent']}%"),
            MetricValue(label="Regions", value=str(s["num_regions"])),
        ]

    elif intent == "CHANGE_DETECTION":
        if optical is None or optical_t2 is None:
            raise ValueError("Change detection requires two optical acquisitions (T1 & T2).")
        params = next((s.parameters for s in intent_result.tool_pipeline
                       if s.tool_name == "tool_bitemporal_change_detection"), {})
        result = change_detection.detect_changes(
            optical, optical_t2, target_class=params.get("target_class")
        )
        engine_name = "bitemporal_change_detection"
        stats["change"] = {**result["stats"], "threshold": result["threshold"]}
        overlays["change_overlay_t2"] = result["overlay_png"]
        s = result["stats"]
        arrow = {"increased": "▲", "decreased": "▼", "stable": "■"}[s["net_direction"]]
        metrics = [
            MetricValue(label="Changed Area", value=f"{s['changed_area_km2']} km²"),
            MetricValue(label="Change", value=f"{s['change_percent']}%"),
            MetricValue(label="Net Signal", value=f"{arrow} {s['net_direction'].title()}"),
            MetricValue(label="Regions", value=str(s["num_change_regions"])),
        ]

    elif intent == "CROSS_MODAL_FUSION":
        if optical is None or sar is None:
            raise ValueError("Cross-modal fusion requires both an optical and a SAR image.")
        params = next((s.parameters for s in intent_result.tool_pipeline
                       if s.tool_name == "tool_cross_modal_fusion"), {})
        result = fusion_engine.fuse_analysis(optical, sar, params.get("query_type"))
        engine_name = "cross_modal_fusion"
        stats["fusion"] = result["stats"]
        overlays["fusion_landcover_overlay"] = result["overlay_png"]
        f = result["stats"]
        cm = f["cross_modal"]
        metrics = [
            MetricValue(label="Water", value=f"{f['water']['area_km2']} km²"),
            MetricValue(label="Built-up", value=f"{f['built_up']['area_km2']} km²"),
            MetricValue(label="Vegetation", value=f"{f['vegetation']['area_km2']} km²"),
            MetricValue(label="SAR Rejections", value=f"{cm['sar_rejection_percent']}% FP"),
        ]

    else:
        rasters = [r for r in (optical, sar) if r is not None]
        if not rasters:
            raise ValueError("VQA requires at least one image.")
        answer_text, vqa_metrics, vqa_source = vqa_engine.answer_query(rasters, user_query)
        engine_name = "vqa_single"
        stats["vqa_source"] = vqa_source
        stats["vqa_metrics"] = vqa_metrics
        overlays["optical_render"] = _png_bytes_from_raster(rasters[0])
        metrics = [
            MetricValue(label="Answer Source", value="Groq Vision" if vqa_source == "groq_vision" else "Rule-based CV"),
            MetricValue(label="Images Analyzed", value=str(len(rasters))),
        ]
        return EngineOutput(
            engine_name=engine_name,
            answer_markdown=answer_text,
            metrics=metrics,
            overlays={k: _png_to_data_url(v) for k, v in overlays.items()},
            raw_stats=stats,
        )

    answer_text, synth_source = synthesize(user_query, intent_result, stats)
    metrics.append(MetricValue(label="Synthesis", value=synth_source))
    return EngineOutput(
        engine_name=engine_name,
        answer_markdown=answer_text,
        metrics=metrics,
        overlays={k: _png_to_data_url(v) for k, v in overlays.items()},
        raw_stats=stats,
    )


def _png_bytes_from_raster(raster: RasterData) -> bytes:
    from src.gis.pipeline import overlay_png, to_rgb_render

    return overlay_png(to_rgb_render(raster), mask=None)
