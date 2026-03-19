""""
PROTECTION – DROITS D’AUTEUR – 2026

 

© 2026 

 

Ce travail (code source + documents associés) est protégé par le droit d’auteur.

Autorisation limitée à la lecture seule pour évaluation du cours uniquement.

Aucune cession de droits. Toute autre utilisation (reproduction, modification,

exploitation pédagogique ou commerciale) interdite sans accord écrit préalable.

"""

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
    """Phase 3 — Pipeline LangGraph multi-modèles."""
    import traceback
    try:
        initial_state = {
            "structured_data": request.structured_data.model_dump(),
            "image_analysis":  request.image_analysis or "",
            "patient_address": getattr(request, "patient_address", "") or "",
            "step_outputs":    {},
            "current_step":    0,
            "diagnostic":      None,
            "prompt_final":    None,
        }
        final_state = await pipeline.ainvoke(initial_state)
        return {
            "step_outputs":  final_state["step_outputs"],
            "diagnostic":    final_state["diagnostic"],
            "prompt_final":  final_state.get("prompt_final", ""),
            "step_4_detail": final_state["step_outputs"].get("step_4_detail", ""),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/prompt")
async def export_prompt(payload: dict):
    """Génère un fichier texte du prompt final + justifications pour téléchargement."""
    from fastapi.responses import Response
    diag     = payload.get("diagnostic", {})
    prompt   = payload.get("prompt_final", "")
    detail   = payload.get("step_4_detail", "")
    patient  = payload.get("structured_data", {})

    lines = [
        "=" * 70,
        "PIPELINE DIAGNOSTIC IA — RAPPORT D'EXPLICABILITÉ",
        "=" * 70,
        "",
        f"Patient : {patient.get('motif_consultation', 'N/A')}",
        f"Urgence : {diag.get('urgence', 'N/A')}",
        f"Score de confiance : {diag.get('score_confiance', 'N/A')}%",
        "",
        "─" * 70,
        "PROMPT FINAL SOUMIS AUX MODÈLES",
        "─" * 70,
        prompt,
        "",
        "─" * 70,
        "DIAGNOSTIC FINAL",
        "─" * 70,
    ]
    for h in diag.get("hypotheses", []):
        lines += [
            f"\nHypothèse #{h.get('rang')} — {h.get('diagnostic')} ({h.get('probabilite')}%)",
            f"CIM-10 : {h.get('cim10', 'N/A')}",
            f"Justification : {h.get('justification', 'N/A')}",
            f"Examens : {', '.join(h.get('examens', []))}",
        ]

    lines += [
        "",
        "─" * 70,
        "RAISONNEMENT EXPLICITE",
        "─" * 70,
        diag.get("raisonnement_explicite", "N/A"),
    ]

    consensus = diag.get("consensus", {})
    if consensus:
        lines += [
            "",
            "─" * 70,
            "CONSENSUS MULTI-MODÈLES",
            "─" * 70,
            f"Accord : {consensus.get('accord_ia', 'N/A')}",
            f"Points d'accord : {consensus.get('point_accord', 'N/A')}",
            f"Points de divergence : {consensus.get('point_divergence', 'N/A')}",
            f"Arbitrage : {consensus.get('arbitrage', 'N/A')}",
            "",
            f"IA A ({consensus.get('avis_ia_a', {}).get('diagnostic_principal', 'N/A')}) — {consensus.get('avis_ia_a', {}).get('probabilite', 'N/A')}%",
            f"  → {consensus.get('avis_ia_a', {}).get('specificite', 'N/A')}",
            "",
            f"IA B ({consensus.get('avis_ia_b', {}).get('diagnostic_principal', 'N/A')}) — {consensus.get('avis_ia_b', {}).get('probabilite', 'N/A')}%",
            f"  → {consensus.get('avis_ia_b', {}).get('specificite', 'N/A')}",
        ]

    if detail:
        try:
            d = json.loads(detail)
            lines += [
                "",
                "─" * 70,
                "DÉTAIL TECHNIQUE DES MODÈLES",
                "─" * 70,
                f"Modèle A : {d.get('ia_a', {}).get('model', 'N/A')}",
                f"Modèle B : {d.get('ia_b', {}).get('model', 'N/A')}",
                f"Arbitre  : {d.get('ia_c', {}).get('model', 'N/A')}",
            ]
        except Exception:
            pass

    lines += [
        "",
        "─" * 70,
        "⚠️  Aide à la décision uniquement.",
        "    Le diagnostic final relève de la responsabilité exclusive du médecin.",
        "─" * 70,
    ]

    content_str = "\n".join(lines)

    return Response(
        content=content_str.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=rapport_diagnostic.txt"},
    )


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