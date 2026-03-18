"""
Pipeline Diagnostic Médical par IA — FastAPI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from models.schemas import (
    PatientRawData, StructuredPatientData,
    PipelineRequest, PipelineResult,
)
from pipeline.graph import build_pipeline_graph

app = FastAPI(title="Pipeline Diagnostic IA", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = build_pipeline_graph()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/extract", response_model=StructuredPatientData)
async def extract_structured_data(raw: PatientRawData):
    """Phase 2 — Extraction et structuration des données brutes."""
    from pipeline.nodes import extract_and_structure
    try:
        return await extract_and_structure(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-document")
async def analyze_document(payload: dict):
    """
    Analyse un document médical ou une image clinique via llama-4-scout (vision).
    Payload : { image_base64, media_type, filename }
    """
    from pipeline.nodes import analyze_document as _analyze
    image_b64  = payload.get("image_base64", "")
    media_type = payload.get("media_type", "image/jpeg")
    filename   = payload.get("filename", "document")

    if not image_b64:
        raise HTTPException(status_code=400, detail="image_base64 requis")
    try:
        result = await _analyze(image_b64, media_type, filename)
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/run")
async def run_diagnostic_pipeline(request: PipelineRequest):
    """Phase 3 — Pipeline LangGraph en 4 nœuds."""
    import traceback
    try:
        initial_state = {
            "structured_data": request.structured_data.model_dump(),
            "image_analysis":  request.image_analysis or "",
            "step_outputs":    {},
            "current_step":    0,
            "diagnostic":      None,
        }
        final_state = await pipeline.ainvoke(initial_state)
        return {
            "step_outputs": final_state["step_outputs"],
            "diagnostic":   final_state["diagnostic"],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/questions/generate")
async def generate_questions(payload: dict):
    """Phase 1 — Génération dynamique des questions diagnostiques."""
    transcription = payload.get("transcription", "")
    if len(transcription) < 40:
        raise HTTPException(status_code=400, detail="Transcription trop courte")
    from pipeline.nodes import generate_diagnostic_questions
    questions = await generate_diagnostic_questions(transcription)
    return {"questions": questions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)