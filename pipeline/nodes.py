"""
Nœuds du Pipeline LangGraph
LLM texte  : Groq llama-3.3-70b-versatile
LLM vision : Groq meta-llama/llama-4-scout-17b-16e-instruct
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
from groq import Groq

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
# Nœuds 1-3 : modèle léger pour économiser les tokens quotidiens
PREP_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"  # Init/Gen/Eval — rapide, quota illimité
# Nœud 4 : multi-LLM (quotas séparés)
TEXT_MODEL    = "llama-3.3-70b-versatile"                    # IA A — quota 100k/j
MODEL_B       = "moonshotai/kimi-k2-instruct"                # IA B — 262k ctx, quota séparé
MODEL_ARBITER = "openai/gpt-oss-120b"                        # IA C — arbitre, quota séparé
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def call_llm(system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024, model: str = None) -> str:
    import time, re
    target_model = model or TEXT_MODEL
    fallback_models = [target_model, PREP_MODEL, "llama-3.1-8b-instant"]
    
    for attempt, m in enumerate(fallback_models):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                # Extraire le temps d'attente si disponible
                wait_match = re.search(r"try again in (\d+)m(\d+\.\d+)s", err_str)
                if wait_match and attempt == 0:
                    wait_sec = int(wait_match.group(1))*60 + float(wait_match.group(2))
                    if wait_sec < 120:  # Si attente < 2min, on attend
                        time.sleep(min(wait_sec + 2, 120))
                        try:
                            client2 = Groq(api_key=GROQ_API_KEY)
                            response = client2.chat.completions.create(
                                model=m, messages=[{"role":"system","content":system},{"role":"user","content":user}],
                                temperature=temperature, max_tokens=max_tokens,
                            )
                            return response.choices[0].message.content
                        except Exception:
                            pass
                if attempt < len(fallback_models) - 1:
                    print(f"[Rate limit] {m} épuisé, bascule vers {fallback_models[attempt+1]}")
                    continue
                raise Exception(f"Tous les modèles sont en rate limit. Réessayez dans quelques minutes.")
            raise  # Autre erreur → on la remonte


def call_vision(image_base64: str, media_type: str, prompt: str) -> str:
    """Analyse une image ou un document via llama-4-scout (multimodal)."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Prompts système ──────────────────────────────────────────────────────────

SYSTEM_INIT = """Tu es l'IA d'initialisation d'un pipeline de diagnostic médical assisté.
Ton rôle est de contextualiser rigoureusement le cas clinique à partir des données structurées.
Tu dois :
1. Synthétiser les éléments cliniques clés
2. Identifier les systèmes organiques impliqués
3. Évaluer le niveau d'urgence préliminaire
4. Intégrer les résultats d'analyses d'images/documents si présents
Sois précis, exhaustif et objectif. Utilise la terminologie médicale appropriée."""

SYSTEM_GEN = """Tu es une IA médicale spécialisée en diagnostic différentiel.
Génère un diagnostic différentiel complet à partir du contexte clinique.
Pour chaque hypothèse :
- Justifie cliniquement en citant les éléments du dossier
- Identifie les drapeaux rouges et signes d'alarme
- Note les éléments manquants qui renforceraient ou excluraient chaque hypothèse
Classe les hypothèses par probabilité décroissante."""

SYSTEM_EVAL = """Tu es un LLM d'évaluation et d'optimisation médicale.
Évalue la qualité du raisonnement diagnostique selon :
- Cohérence clinique et logique médicale
- Exhaustivité du diagnostic différentiel
- Pertinence des drapeaux rouges
- Qualité de l'argumentation
Identifie les lacunes et produis un prompt diagnostique optimisé pour l'étape prédictive."""

SYSTEM_PRED = """Tu es l'IA médicale prédictive finale d'un pipeline de diagnostic assisté.

RÈGLE SUR LE NOMBRE D'HYPOTHÈSES :
- 1 hypothèse si un diagnostic est clairement dominant (>75%) et les autres peu plausibles
- 2 hypothèses si deux diagnostics sont sérieusement à considérer
- 3 hypothèses maximum sinon. Ne force JAMAIS 3 si non justifié cliniquement.

RÈGLE SUR LA CONDUITE À SUIVRE — choisis UN seul niveau :
- "maison" : symptômes bénins, pas d'examen urgent nécessaire
- "laboratoire" : examens complémentaires ambulatoires nécessaires
- "hopital" : urgence vitale ou plateau technique hospitalier requis

RÈGLES STRICTES PAR NIVEAU pour remplir les champs de conduite :
- niveau "maison" : remplis conseils_maison (conseils pratiques + signes d'aggravation à surveiller). NE PAS remplir actes_labo, raison_hospitalisation, en_attendant_hopital.
- niveau "laboratoire" : remplis actes_labo (examens à réaliser avec motif) + conseils_maison (surveillance en attendant les résultats). NE PAS remplir raison_hospitalisation, en_attendant_hopital. Les instructions_immediates doivent concerner UNIQUEMENT la démarche diagnostique (ex: "Se rendre au laboratoire dès aujourd'hui", "Apporter l'ordonnance jointe").
- niveau "hopital" : remplis raison_hospitalisation (motif précis de l'envoi) + en_attendant_hopital (que faire en attendant les secours/transport). NE PAS remplir actes_labo. Les instructions_immediates doivent être des gestes d'urgence immédiats.

Réponds UNIQUEMENT en JSON valide, sans markdown ni texte supplémentaire."""


# ─── Nœuds du pipeline ───────────────────────────────────────────────────────

async def initialization_node(state: dict) -> dict:
    """Nœud ① — Initialisation (PI)"""
    ctx = json.dumps(state["structured_data"], ensure_ascii=False, indent=2)
    image_analysis = state.get("image_analysis", "")

    image_section = ""
    if image_analysis:
        image_section = f"\n\nAnalyse des documents/images fournis :\n{image_analysis}"

    output = call_llm(
        system=SYSTEM_INIT,
        model=PREP_MODEL,
        user=f"""Données patient structurées :
{ctx}{image_section}

Initialise le contexte clinique pour le pipeline de diagnostic.
Synthétise les éléments clés et prépare le raisonnement.""",
        temperature=0.2,
    )

    return {
        **state,
        "current_step": 1,
        "step_outputs": {**state.get("step_outputs", {}), "step_1": output},
    }


async def generation_node(state: dict) -> dict:
    """Nœud ② — Génération diagnostique (IA₁)"""
    step1_output = state["step_outputs"]["step_1"]

    output = call_llm(
        system=SYSTEM_GEN,
        model=PREP_MODEL,
        user=f"""{step1_output}

Génère un diagnostic différentiel complet avec justification clinique,
drapeaux rouges, et éléments manquants pour chaque hypothèse.""",
        temperature=0.4,
    )

    return {
        **state,
        "current_step": 2,
        "step_outputs": {**state["step_outputs"], "step_2": output},
    }


async def evaluation_node(state: dict) -> dict:
    """Nœud ③ — Évaluation LLM (IA₂)"""
    step2_output = state["step_outputs"]["step_2"]

    output = call_llm(
        system=SYSTEM_EVAL,
        model=PREP_MODEL,
        user=f"""Raisonnement diagnostique à évaluer :
{step2_output}

Évalue ce raisonnement et produis un prompt diagnostique optimisé
pour l'étape prédictive finale.""",
        temperature=0.3,
    )

    return {
        **state,
        "current_step": 3,
        "step_outputs": {**state["step_outputs"], "step_3": output},
    }


def _parse_diag(raw: str) -> dict | None:
    """Parse JSON diagnostique depuis la réponse LLM."""
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        start, end = clean.find("{"), clean.rfind("}") + 1
        return json.loads(clean[start:end])
    except (json.JSONDecodeError, ValueError):
        return None

def _fallback_diag() -> dict:
    return {
        "hypotheses": [{"rang":1,"diagnostic":"À déterminer","probabilite":60,
            "cim10":"R00-R99","justification":"Données insuffisantes","examens":["Bilan biologique complet"]}],
        "urgence":"moyenne","score_confiance":50,
        "conduite":{"niveau":"laboratoire","raison":"Bilan complémentaire nécessaire",
            "instructions_immediates":[],"actes_labo":["NFS, CRP, bilan métabolique"],"conseils_maison":[]},
    }

async def predictive_node(state: dict) -> dict:
    """
    Nœud ④ — IA Prédictive Multi-Modèles (IA₃)
    
    Architecture :
      IA A (llama-3.3-70b)       ─┐
                                  ├─→ IA C (gpt-oss-120b) arbitre → Diagnostic final
      IA B (kimi-k2-instruct-0905) ─┘
    """
    step3_output = state["step_outputs"]["step_3"]
    ctx = json.dumps(state["structured_data"], ensure_ascii=False, indent=2)

    # Prompt partagé — identique pour A et B
    SHARED_PROMPT = f"""Prompt diagnostique optimisé :
{step3_output}

Données patient originales :
{ctx}

INSTRUCTIONS CRITIQUES SUR LES HYPOTHÈSES :
- Toujours proposer PLUSIEURS hypothèses (idéalement 2-3) même si l'une domine
- Chaque hypothèse doit être cliniquement défendable, pas juste un remplissage
- Les hypothèses en désaccord entre IA sont précieuses pour l'arbitrage — n'hésitez pas à défendre votre point de vue
- Si un diagnostic alternatif est sérieux, mettez-le même à 20-30%, il compte
- Justifiez en citant les signes SPÉCIFIQUES qui penchent pour ou contre

Génère le diagnostic. Réponds UNIQUEMENT en JSON valide :
{{
  "hypotheses": [
    {{"rang":1,"diagnostic":"...","probabilite":65,"cim10":"...","justification":"Justification de 2-3 phrases citant les signes cliniques clés pour et contre","examens":["..."]}},
    {{"rang":2,"diagnostic":"...","probabilite":25,"cim10":"...","justification":"Pourquoi ce diagnostic alternatif doit rester dans le différentiel","examens":["..."]}},
    {{"rang":3,"diagnostic":"...","probabilite":10,"cim10":"...","justification":"Diagnostic à éliminer formellement","examens":["..."]}}
  ],
  "urgence":"haute|moyenne|faible",
  "score_confiance":85,
  "conduite":{{
    "niveau":"maison|laboratoire|hopital",
    "raison":"...",
    "instructions_immediates":["..."],
    "conseils_maison":["..."],
    "actes_labo":["..."],
    "raison_hospitalisation":"...",
    "en_attendant_hopital":["..."]
  }},
  "raisonnement_explicite": "Expliquer en 3-4 phrases POURQUOI ce diagnostic a été retenu : quels éléments cliniques sont déterminants, quels diagnostics ont été écartés et pourquoi"
}}"""

    # ── Appel IA A & B en parallèle (asyncio + executor) ────────────────────
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        future_a = loop.run_in_executor(executor, lambda: call_llm(SYSTEM_PRED, SHARED_PROMPT, 0.15, 1800, TEXT_MODEL))
        future_b = loop.run_in_executor(executor, lambda: call_llm(SYSTEM_PRED, SHARED_PROMPT, 0.25, 1800, MODEL_B))
        raw_a, raw_b = await asyncio.gather(future_a, future_b)
    finally:
        executor.shutdown(wait=False)

    diag_a = _parse_diag(raw_a)
    diag_b = _parse_diag(raw_b)

    # ── IA C — Arbitre et synthèse ───────────────────────────────────────────
    SYSTEM_ARBITER = """Tu es un médecin expert arbitre dans un système de diagnostic multi-IA.
Tu reçois les diagnostics indépendants de deux IA médicales et tu dois :
1. Analyser les convergences et divergences
2. Identifier le diagnostic le plus cliniquement solide
3. Produire le diagnostic final de consensus
4. Expliquer explicitement pourquoi tu as tranché en faveur de tel ou tel modèle

Sois rigoureux, honnête sur les incertitudes, et cite les éléments cliniques qui ont guidé ton arbitrage.
Réponds UNIQUEMENT en JSON valide, sans markdown."""

    arbiter_prompt = f"""Diagnostic IA A ({TEXT_MODEL}) :
{raw_a[:2000]}

Diagnostic IA B ({MODEL_B}) :
{raw_b[:2000]}

Données patient originales :
{ctx}

Analyse ces deux diagnostics et produis le consensus final.
Réponds UNIQUEMENT avec ce JSON :
{{
  "hypotheses": [
    {{"rang":1,"diagnostic":"...","probabilite":65,"cim10":"...","justification":"Justification détaillée citant les éléments cliniques clés","examens":["..."]}}
  ],
  "urgence":"haute|moyenne|faible",
  "score_confiance":85,
  "conduite":{{
    "niveau":"maison|laboratoire|hopital",
    "raison":"...",
    "instructions_immediates":[],
    "conseils_maison":[],
    "actes_labo":[],
    "raison_hospitalisation":"",
    "en_attendant_hopital":[]
  }},
  "raisonnement_explicite": "Pourquoi ce diagnostic final : éléments cliniques déterminants, diagnostics écartés",
  "consensus": {{
    "accord_ia": "total|partiel|divergence",
    "point_accord": "Sur quoi A et B étaient d'accord",
    "point_divergence": "En quoi A et B différaient",
    "arbitrage": "Pourquoi l'arbitre a tranché en faveur de cette conclusion",
    "avis_ia_a": {{
      "diagnostic_principal": "Diagnostic principal de A",
      "probabilite": 60,
      "specificite": "Ce que A avait de spécifique ou de pertinent"
    }},
    "avis_ia_b": {{
      "diagnostic_principal": "Diagnostic principal de B",
      "probabilite": 55,
      "specificite": "Ce que B avait de spécifique ou de pertinent"
    }}
  }}
}}"""

    raw_c = call_llm(SYSTEM_ARBITER, arbiter_prompt, 0.1, 2200, MODEL_ARBITER)
    diag_final = _parse_diag(raw_c) or diag_a or diag_b or _fallback_diag()

    # Assurer les champs d'explicabilité même si absents
    if "raisonnement_explicite" not in diag_final:
        diag_final["raisonnement_explicite"] = "Diagnostic établi par consensus multi-modèles."
    if "consensus" not in diag_final:
        diag_final["consensus"] = {
            "accord_ia": "partiel",
            "point_accord": "Orientation diagnostique générale",
            "point_divergence": "Pondération des hypothèses",
            "arbitrage": "Consensus retenu sur les éléments cliniques les plus saillants",
            "avis_ia_a": {"diagnostic_principal": diag_a["hypotheses"][0]["diagnostic"] if diag_a else "N/A", "probabilite": diag_a["hypotheses"][0]["probabilite"] if diag_a else 50, "specificite": "Avis llama-3.3-70b"},
            "avis_ia_b": {"diagnostic_principal": diag_b["hypotheses"][0]["diagnostic"] if diag_b else "N/A", "probabilite": diag_b["hypotheses"][0]["probabilite"] if diag_b else 50, "specificite": "Avis kimi-k2-instruct"},
        }

    # Stocker le prompt final et les outputs bruts pour téléchargement
    prompt_final = SHARED_PROMPT
    step4_detail = {
        "prompt_final": prompt_final,
        "ia_a": {"model": TEXT_MODEL, "raw": raw_a, "parsed": diag_a},
        "ia_b": {"model": MODEL_B,   "raw": raw_b, "parsed": diag_b},
        "ia_c": {"model": MODEL_ARBITER, "raw": raw_c, "role": "arbitre"},
    }

    return {
        **state,
        "current_step": 4,
        "step_outputs": {
            **state["step_outputs"],
            "step_4": raw_c,
            "step_4_detail": json.dumps(step4_detail, ensure_ascii=False, indent=2),
        },
        "diagnostic": diag_final,
        "prompt_final": prompt_final,
    }


# ─── Fonctions utilitaires ────────────────────────────────────────────────────

async def analyze_document(image_base64: str, media_type: str, filename: str) -> str:
    """
    Analyse un document médical ou une image clinique via llama-4-scout.
    Appelée par l'endpoint /api/analyze-document.
    """
    file_type = "radiographie/image médicale" if media_type.startswith("image/") else "document médical (PDF/scan)"

    prompt = f"""Analyse ce {file_type} ({filename}) dans un contexte médical clinique.
Extrais et synthétise :
- Tous les résultats, valeurs, mesures visibles
- Anomalies, signes pathologiques identifiables
- Conclusions ou interprétations présentes dans le document
- Tout élément cliniquement pertinent pour le diagnostic

Sois précis et exhaustif. Si c'est une image radiologique, décris les findings.
Si c'est un bilan biologique, liste les valeurs anormales avec les normes."""

    try:
        return call_vision(image_base64, media_type, prompt)
    except Exception as e:
        return f"[Analyse indisponible : {str(e)}]"


async def extract_and_structure(raw_data) -> dict:
    """Phase 2 — Extraction et structuration."""
    data_str = json.dumps({
        "transcription":       raw_data.transcription,
        "constantes":          raw_data.vital_signs.model_dump(),
        "symptomes_confirmes": raw_data.confirmed_symptoms,
    }, ensure_ascii=False, indent=2)

    try:
        output = call_llm(
            system="Tu es une IA d'extraction médicale. Retourne uniquement un JSON valide, sans markdown.",
            user=f"""Extrais et normalise ces données patient :
{data_str}

Réponds UNIQUEMENT en JSON valide :
{{
  "motif_consultation": "...",
  "symptomes_principaux": ["..."],
  "constantes_vitales": {{}},
  "antecedents": "...",
  "symptomes_confirmes": ["..."],
  "urgence_estimee": "haute|moyenne|faible",
  "contexte_clinique": "...",
  "score_gravite": 7
}}""",
            temperature=0.1,
        )
        clean = output.replace("```json", "").replace("```", "").strip()
        start, end = clean.find("{"), clean.rfind("}") + 1
        return json.loads(clean[start:end])
    except Exception:
        return {
            "motif_consultation":   raw_data.transcription[:200],
            "symptomes_principaux": raw_data.confirmed_symptoms,
            "constantes_vitales":   raw_data.vital_signs.model_dump(),
            "antecedents":          None,
            "symptomes_confirmes":  raw_data.confirmed_symptoms,
            "urgence_estimee":      "moyenne",
            "contexte_clinique":    raw_data.transcription,
            "score_gravite":        5,
        }


async def generate_diagnostic_questions(transcription: str) -> list[str]:
    """Phase 1 — Génération dynamique des questions."""
    try:
        output = call_llm(
            system="Tu es un assistant médical. Réponds uniquement en JSON array valide, sans markdown.",
            user=f"""Transcription : "{transcription[:600]}"
Génère exactement 8 questions diagnostiques cliniques en français.
Réponds UNIQUEMENT avec un JSON array : ["Question 1 ?", ...]""",
            temperature=0.5,
        )
        text = output.strip()
        start, end = text.find("["), text.rfind("]") + 1
        questions = json.loads(text[start:end])
        return questions[:8] if isinstance(questions, list) else []
    except Exception:
        return [
            "Antécédents médicaux significatifs ?",
            "Début et évolution des symptômes ?",
            "La douleur est-elle irradiante ?",
            "Facteurs de risque cardiovasculaire ?",
            "Médicaments en cours ?", "Allergies connues ?",
            "Fièvre ou frissons associés ?", "Épisodes similaires ?",
        ]