import json
import re

import config
from src.orchestrator.schemas import IntentResult, QueryContext, ToolStep, VALID_INTENTS

ORCHESTRATOR_SYSTEM_PROMPT = """You are SatQuery AI Orchestrator, an intelligent routing agent for ISRO's satellite remote-sensing analysis platform. Analyze the user's natural language query, extract spatio-temporal intent, and select the optimal specialist execution tool.

AVAILABLE SPECIALIST TOOLS:
1. tool_vqa_single(optical_image_id, prompt)
   - Use when user asks to describe, detect, or query a single satellite image (e.g., "Describe land cover", "Is there a highway in this image?").
2. tool_cross_modal_fusion(optical_image_id, sar_image_id, query_type)
   - Use when user query explicitly mentions fusing Optical (Sentinel-2) and SAR (Sentinel-1) data or identifying features resistant to cloud cover / water visibility.
3. tool_bitemporal_change_detection(image_before_id, image_after_id, target_class)
   - Use when user asks for temporal comparison (e.g., "What changed between T1 and T2?", "Has the urban area expanded?").
4. tool_spatial_segmentation(image_id, target_feature)
   - Use when user asks to highlight, mask, or outline specific features (e.g., "Highlight water bodies", "Mask all agricultural zones").

INPUT CONTEXT: a JSON object with user_query, has_optical, has_sar, is_bitemporal.

RULES:
- If two optical images are provided and the query compares time periods, choose CHANGE_DETECTION.
- If both SAR and Optical are provided and the query asks for joint analysis, choose CROSS_MODAL_FUSION.
- If the query asks to highlight/mask/segment/outline/map a specific feature class, choose SPATIAL_SEGMENTATION.
- Otherwise default to VQA_SINGLE.

Respond with JSON ONLY in this exact schema:
{
  "primary_intent": "VQA_SINGLE" | "CROSS_MODAL_FUSION" | "CHANGE_DETECTION" | "SPATIAL_SEGMENTATION",
  "reasoning": "Brief technical explanation of tool selection",
  "tool_pipeline": [
    {"step": 1, "tool_name": "string", "parameters": {}}
  ],
  "requires_change_map": true | false
}"""

_CHANGE_KEYWORDS = [
    "change", "changed", "between t1", "t1 and t2", "before and after",
    "temporal", "expansion", "expanded", "shrunk", "shrink", "growth",
    "deforest", "over time", "compared", "comparison", "difference between",
]
_FUSION_KEYWORDS = [
    "sar", "fusion", "fuse", "combined", "combine", "both sensors",
    "cloud-penetrating", "radar", "backscatter", "vv", "vh", "cross-modal",
    "cross modal",
]
_SEGMENT_KEYWORDS = [
    "highlight", "mask", "segment", "outline", "map all", "show me all",
    "mark all", "draw", "locate all", "identify all", "overlay",
]


def _heuristic_route(ctx: QueryContext) -> IntentResult:
    q = ctx.user_query.lower()

    def kw(words):
        return any(w in q for w in words)

    if ctx.has_sar and ctx.has_optical and kw(_FUSION_KEYWORDS):
        intent = "CROSS_MODAL_FUSION"
        reasoning = (
            "Query requests joint Sentinel-1 SAR + Sentinel-2 optical analysis; "
            "routing to cross-modal fusion engine."
        )
        pipeline = [ToolStep(step=1, tool_name="tool_cross_modal_fusion",
                             parameters={"query_type": _extract_target(q) or "land_cover"})]
        return IntentResult(primary_intent=intent, reasoning=reasoning,
                            tool_pipeline=pipeline, requires_change_map=False, source="heuristic")

    if ctx.is_bitemporal or kw(_CHANGE_KEYWORDS):
        intent = "CHANGE_DETECTION"
        reasoning = (
            "Two optical acquisitions supplied and the query references temporal comparison; "
            "routing to bi-temporal change detection."
        )
        pipeline = [ToolStep(step=1, tool_name="tool_bitemporal_change_detection",
                             parameters={"target_class": _extract_target(q)})]
        return IntentResult(primary_intent=intent, reasoning=reasoning,
                            tool_pipeline=pipeline, requires_change_map=True, source="heuristic")

    if kw(_SEGMENT_KEYWORDS):
        target = _extract_target(q) or "water"
        reasoning = (
            "Query requests highlighting/masking of a specific feature class; "
            "routing to spatial segmentation."
        )
        pipeline = [ToolStep(step=1, tool_name="tool_spatial_segmentation",
                             parameters={"target_feature": target})]
        return IntentResult(primary_intent="SPATIAL_SEGMENTATION", reasoning=reasoning,
                            tool_pipeline=pipeline, requires_change_map=False, source="heuristic")

    reasoning = "General description/detection request on a single acquisition; routing to VQA."
    pipeline = [ToolStep(step=1, tool_name="tool_vqa_single", parameters={"prompt": ctx.user_query})]
    return IntentResult(primary_intent="VQA_SINGLE", reasoning=reasoning,
                        tool_pipeline=pipeline, requires_change_map=False, source="heuristic")


_TARGET_MAP = {
    "water": ["water", "river", "lake", "reservoir", "pond", "coast"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "agricultur", "farm", "green"],
    "built_up": ["built-up", "built up", "urban", "building", "house", "settlement", "infrastructure", "road", "highway"],
    "cloud": ["cloud"],
}


def _extract_target(q: str) -> str:
    ql = q.lower()
    best, best_pos = "", 10**9
    for target, words in _TARGET_MAP.items():
        for w in words:
            pos = ql.find(w)
            if pos != -1 and pos < best_pos:
                best, best_pos = target, pos
    return best


def _parse_llm_json(raw: str, ctx: QueryContext) -> IntentResult:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    intent = str(data.get("primary_intent", "")).strip().upper()
    if intent not in VALID_INTENTS:
        raise ValueError(f"invalid intent: {intent}")
    steps_raw = data.get("tool_pipeline") or []
    steps = []
    for i, s in enumerate(steps_raw, start=1):
        if isinstance(s, dict):
            steps.append(ToolStep(
                step=int(s.get("step", i)),
                tool_name=str(s.get("tool_name", "")),
                parameters=s.get("parameters") or {},
            ))
    if not steps:
        steps = [_default_step(intent, ctx)]
    params = steps[0].parameters
    if intent == "SPATIAL_SEGMENTATION" and not params.get("target_feature"):
        params["target_feature"] = _extract_target(ctx.user_query.lower()) or "water"
    if intent == "CHANGE_DETECTION" and not params.get("target_class"):
        params["target_class"] = _extract_target(ctx.user_query.lower())
    return IntentResult(
        primary_intent=intent,
        reasoning=str(data.get("reasoning", "")),
        tool_pipeline=steps,
        requires_change_map=bool(data.get("requires_change_map", intent == "CHANGE_DETECTION")),
        source="llm",
    )


def _default_step(intent: str, ctx: QueryContext) -> ToolStep:
    target = _extract_target(ctx.user_query.lower())
    if intent == "CHANGE_DETECTION":
        return ToolStep(step=1, tool_name="tool_bitemporal_change_detection",
                        parameters={"target_class": target})
    if intent == "CROSS_MODAL_FUSION":
        return ToolStep(step=1, tool_name="tool_cross_modal_fusion",
                        parameters={"query_type": target or "land_cover"})
    if intent == "SPATIAL_SEGMENTATION":
        return ToolStep(step=1, tool_name="tool_spatial_segmentation",
                        parameters={"target_feature": target or "water"})
    return ToolStep(step=1, tool_name="tool_vqa_single", parameters={"prompt": ctx.user_query})


def route_query(ctx: QueryContext) -> IntentResult:
    if config.llm_available():
        try:
            from groq import Groq

            client = Groq(api_key=config.GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=config.GROQ_TEXT_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps({
                            "user_query": ctx.user_query,
                            "has_optical": ctx.has_optical,
                            "has_sar": ctx.has_sar,
                            "is_bitemporal": ctx.is_bitemporal,
                        }),
                    },
                ],
            )
            raw = completion.choices[0].message.content or ""
            return _parse_llm_json(raw, ctx)
        except Exception:
            pass
    return _heuristic_route(ctx)
