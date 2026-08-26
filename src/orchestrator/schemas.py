from typing import Literal, Optional
from pydantic import BaseModel, Field

Intent = Literal[
    "VQA_SINGLE",
    "CROSS_MODAL_FUSION",
    "CHANGE_DETECTION",
    "SPATIAL_SEGMENTATION",
]

VALID_INTENTS = ("VQA_SINGLE", "CROSS_MODAL_FUSION", "CHANGE_DETECTION", "SPATIAL_SEGMENTATION")


class ToolStep(BaseModel):
    step: int
    tool_name: str
    parameters: dict = Field(default_factory=dict)


class IntentResult(BaseModel):
    primary_intent: Intent
    reasoning: str = ""
    tool_pipeline: list[ToolStep] = Field(default_factory=list)
    requires_change_map: bool = False
    source: Literal["llm", "heuristic"] = "llm"


class QueryContext(BaseModel):
    user_query: str
    has_optical: bool = False
    has_sar: bool = False
    is_bitemporal: bool = False


class MetricValue(BaseModel):
    label: str
    value: str
    help_text: Optional[str] = None


class EngineOutput(BaseModel):
    engine_name: str
    answer_markdown: str
    metrics: list[MetricValue] = Field(default_factory=list)
    overlays: dict[str, str] = Field(default_factory=dict)
    raw_stats: dict = Field(default_factory=dict)
