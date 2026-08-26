import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.router import _heuristic_route
from src.orchestrator.schemas import QueryContext


def q(text, opt=True, sar=False, bi=False):
    return QueryContext(user_query=text, has_optical=opt, has_sar=sar, is_bitemporal=bi)


def test_vqa_default():
    res = _heuristic_route(q("Describe the land cover in this image"))
    assert res.primary_intent == "VQA_SINGLE"
    assert res.tool_pipeline[0].tool_name == "tool_vqa_single"


def test_segmentation_routing():
    res = _heuristic_route(q("Highlight water bodies"))
    assert res.primary_intent == "SPATIAL_SEGMENTATION"
    assert res.tool_pipeline[0].parameters["target_feature"] == "water"

    res2 = _heuristic_route(q("Mask all agricultural zones"))
    assert res2.primary_intent == "SPATIAL_SEGMENTATION"
    assert res2.tool_pipeline[0].parameters["target_feature"] == "vegetation"


def test_change_detection_routing():
    res = _heuristic_route(q("Has urban area expanded between the two dates?", bi=True))
    assert res.primary_intent == "CHANGE_DETECTION"
    assert res.requires_change_map is True
    assert res.tool_pipeline[0].parameters["target_class"] == "built_up"


def test_fusion_routing():
    res = _heuristic_route(
        q("Use the optical and SAR images together to identify built-up and water-covered regions",
          sar=True)
    )
    assert res.primary_intent == "CROSS_MODAL_FUSION"


def test_fusion_priority_over_bitemporal():
    res = _heuristic_route(
        q("Fuse SAR and optical data to analyze water", sar=True, bi=True)
    )
    assert res.primary_intent == "CROSS_MODAL_FUSION"


def test_target_extraction_order():
    res = _heuristic_route(q("Highlight rivers and forests"))
    assert res.tool_pipeline[0].parameters["target_feature"] == "water"
