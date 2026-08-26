from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.gis.pipeline import load_geotiff
from src.orchestrator.router import route_query
from src.orchestrator.schemas import QueryContext
from src.tools.registry import execute_pipeline

app = FastAPI(
    title="SatQuery AI API",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing Analysis (ISRO / SIH26167)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_available": config.llm_available(),
        "text_model": config.GROQ_TEXT_MODEL if config.llm_available() else None,
        "vision_model": config.GROQ_VISION_MODEL if config.llm_available() else None,
    }


@app.post("/api/query")
async def query(
    query_text: str = Form(..., alias="query"),
    optical_file: UploadFile | None = File(None),
    sar_file: UploadFile | None = File(None),
    bitemporal_file: UploadFile | None = File(None),
):
    tmp_dir = Path(tempfile.mkdtemp(prefix="satquery_"))
    try:
        saved = {}

        async def save(upload: UploadFile | None, name: str) -> Path | None:
            if upload is None or not upload.filename:
                return None
            dest = tmp_dir / f"{name}_{Path(upload.filename).name}"
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            return dest

        p_optical = await save(optical_file, "optical")
        p_sar = await save(sar_file, "sar")
        p_t2 = await save(bitemporal_file, "optical_t2")

        optical = load_geotiff(str(p_optical)) if p_optical else None
        sar = load_geotiff(str(p_sar)) if p_sar else None
        optical_t2 = load_geotiff(str(p_t2)) if p_t2 else None

        ctx = QueryContext(
            user_query=query_text,
            has_optical=optical is not None,
            has_sar=sar is not None,
            is_bitemporal=optical_t2 is not None,
        )
        intent_result = route_query(ctx)

        try:
            output = execute_pipeline(intent_result, optical=optical, sar=sar,
                                      optical_t2=optical_t2, user_query=query_text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "intent": intent_result.primary_intent,
            "reasoning": intent_result.reasoning,
            "router_source": intent_result.source,
            "tool_pipeline": [s.model_dump() for s in intent_result.tool_pipeline],
            "requires_change_map": intent_result.requires_change_map,
            "engine": output.engine_name,
            "answer_markdown": output.answer_markdown,
            "metrics": [m.model_dump() for m in output.metrics],
            "overlays_b64": {k: v.split(",", 1)[-1] for k, v in output.overlays.items()},
            "stats": output.raw_stats,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
