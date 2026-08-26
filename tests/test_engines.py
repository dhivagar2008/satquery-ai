import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.gis.pipeline import load_geotiff
from src.gis.synthetic import generate_pair


@pytest.fixture(scope="module")
def rasters(tmp_path_factory):
    out = tmp_path_factory.mktemp("eng")
    paths = generate_pair(str(out / "t1.tif"), str(out / "t2.tif"), str(out / "sar.tif"))
    return {
        "t1": load_geotiff(paths["optical_t1"]),
        "t2": load_geotiff(paths["optical_t2"]),
        "sar": load_geotiff(paths["sar"]),
    }


def test_change_detection_finds_changes(rasters):
    from src.engines.change_detection import detect_changes

    result = detect_changes(rasters["t1"], rasters["t2"])
    s = result["stats"]
    assert s["changed_area_km2"] >= 0
    assert s["change_percent"] < 60
    assert s["net_direction"] in ("increased", "decreased", "stable")
    assert result["overlay_png"].startswith(b"\x89PNG")


def test_segmentation_features(rasters):
    from src.engines.segmentation import canonical_feature, segment_feature

    assert canonical_feature("highlight all lakes") == "water"
    assert canonical_feature("mask agricultural zones") == "vegetation"
    assert canonical_feature("urban buildings") == "built_up"

    res = segment_feature(rasters["t1"], None)
    assert res["feature"] == "water"
    for feat in ("vegetation", "built_up"):
        r = segment_feature(rasters["t1"], feat)
        assert r["stats"]["pixels"] >= 0


def test_fusion_produces_classes(rasters):
    from src.engines.fusion_engine import fuse_analysis

    result = fuse_analysis(rasters["t2"], rasters["sar"], "land_cover")
    f = result["stats"]
    assert set(f.keys()) >= {"water", "built_up", "vegetation", "cross_modal"}
    cm = f["cross_modal"]
    assert cm["fused_water_pixels"] <= cm["optical_only_water_pixels"]
    assert result["overlay_png"].startswith(b"\x89PNG")


def test_vqa_rule_based_fallback(rasters):
    from config import llm_available
    from src.engines.vqa_engine import answer_query

    answer, metrics, source = answer_query([rasters["t1"]], "Describe land cover")
    assert len(answer) > 40
    if not llm_available():
        assert source == "rule_based_cv"


def test_registry_end_to_end(rasters):
    from src.orchestrator.schemas import IntentResult, ToolStep
    from src.tools.registry import execute_pipeline

    intent = IntentResult(
        primary_intent="SPATIAL_SEGMENTATION",
        reasoning="test",
        tool_pipeline=[ToolStep(step=1, tool_name="tool_spatial_segmentation",
                                parameters={"target_feature": "water"})],
        requires_change_map=False,
        source="heuristic",
    )
    out = execute_pipeline(intent, optical=rasters["t1"], user_query="Highlight water bodies")
    assert "segmentation" in out.engine_name
    assert out.answer_markdown
    assert list(out.overlays.values())[0].startswith("data:image/png;base64,")

    intent_change = IntentResult(
        primary_intent="CHANGE_DETECTION",
        reasoning="test",
        tool_pipeline=[ToolStep(step=1, tool_name="tool_bitemporal_change_detection",
                                parameters={"target_class": "built_up"})],
        requires_change_map=True,
        source="heuristic",
    )
    out2 = execute_pipeline(intent_change, optical=rasters["t1"], optical_t2=rasters["t2"],
                            user_query="What changed?")
    assert out2.engine_name == "bitemporal_change_detection"

    with pytest.raises(ValueError):
        execute_pipeline(intent_change, optical=None, user_query="x")
