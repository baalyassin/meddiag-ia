"""

PROTECTION – DROITS D’AUTEUR – 2026

 

© 2026

 

Ce travail (code source + documents associés) est protégé par le droit d’auteur.

Autorisation limitée à la lecture seule pour évaluation du cours uniquement.

Aucune cession de droits. Toute autre utilisation (reproduction, modification,

exploitation pédagogique ou commerciale) interdite sans accord écrit préalable.

"""

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
MODEL_B       = "llama-3.1-8b-instant"                       # IA B — quota séparé
MODEL_ARBITER = "meta-llama/llama-4-scout-17b-16e-instruct"  # IA C — arbitre, quota séparé
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

SYSTEM_PRED = """Tu es un médecin urgentiste senior IA dans un pipeline de diagnostic assisté.
Tu raisonnes comme un clinicien aux urgences : tu priorises l'action sur la description.

══════════════════════════════════════════════════════════════
RÈGLE 0 — IDENTIFICATION DU SYSTÈME ORGANE PRINCIPAL
══════════════════════════════════════════════════════════════
AVANT TOUT, identifie le système principal affecté :
  → RESPIRATOIRE/PULMONAIRE : dyspnée, sibilants, toux productive, SpO2 basse, BPCO, pneumonie
  → CARDIAQUE : douleur thoracique oppressante, ST+, palpitations, OAP, syncope
  → NEUROLOGIQUE : déficit focal, AVC, céphalée brutale, trouble conscience
  → DIGESTIF : douleur abdominale, vomissements, hémorragie digestive
  → INFECTIEUX/SEPSIS : fièvre, frissons, signes de choc, foyer infectieux

⚠️ PIÈGE CRITIQUE À ÉVITER :
- "Insuffisance grave" ≠ "insuffisance cardiaque" si le contexte est respiratoire
- BPCO + hypoxémie → PNEUMOLOGIE, pas cardiologie interventionnelle
- Dyspnée + sibilants + hypercapnie → bronchodilatateurs, PAS diurétiques ni nitrates
- Signes respiratoires isolés → unité de pneumologie ou réanimation respiratoire
- Ne jamais orienter vers cardiologie si aucun signe cardiaque direct (ST+, troponine, BNP)

Ce système dominant DOIT guider : traitement, examens ET orientation spécialisée.

══════════════════════════════════════════════════════════════
RÈGLE 1 — URGENCE VITALE : DÉTECTION ET ANNONCE IMMÉDIATE
══════════════════════════════════════════════════════════════
Avant TOUT, demande-toi : "Ce patient peut-il mourir dans les 60 prochaines minutes ?"
Patterns à détecter PAR SYSTÈME :

RESPIRATOIRE  : SpO2 <88% + détresse → insuffisance respiratoire aiguë → O2 contrôlé + SMUR
CARDIAQUE     : Douleur thoracique + neurovégétatif + ST+ → STEMI → coro <90min (PCI)
NEUROLOGIQUE  : Déficit neurologique brutal → AVC → UNV <4h30 → thrombolyse si éligible
INFECTIEUX    : Fièvre + hypotension + confusion → sepsis → urgences + réa
HÉMORRAGIQUE  : Instabilité + saignement → choc → bloc opératoire

→ urgence="haute" + score_confiance peut descendre à 60-70% (acceptable en urgence)
→ Message PRINCIPAL = "URGENCE [SYSTÈME] → action dans les 10 prochaines minutes"
→ NE PAS diluer avec différentiel long avant l'action

══════════════════════════════════════════════════════════════
RÈGLE 2 — TRAITEMENT : PRÉCISION PAR SYSTÈME, PAS GÉNÉRIQUE
══════════════════════════════════════════════════════════════
Pour chaque traitement proposé, tu DOIS préciser :
  1. Pourquoi il est indiqué pour CE patient (indication spécifique)
  2. Pourquoi il pourrait être dangereux si mal appliqué (contre-indication contextuelle)
  3. L'objectif thérapeutique précis (ex: "SpO2 cible 88-92% en BPCO, pas 98%")

EXEMPLES PAR SYSTÈME :
► RESPIRATOIRE/BPCO exacerbée :
  - Oxygène CONTRÔLÉ : objectif SpO2 88-92% (pas plus → risque hypercapnie)
  - Salbutamol 2.5mg nébulisé ± ipratropium 0.5mg → bronchodilatation
  - Prednisolone 40mg/j PO 5j → réduction inflammation bronchique
  - Antibiotique si expectorations purulentes + fièvre : amoxicilline-acide clavulanique 1g x3/j
  - PAS de diurétiques, PAS de nitrates, PAS de bêtabloquants sauf bradycardie

► CARDIAQUE/STEMI :
  - O2 si SpO2<94%, monitorage, VVP
  - DAPT : aspirine 250mg IV + ticagrélor 180mg PO
  - Anticoagulation : héparine non fractionnée 60UI/kg IV
  - Appel cardiologie interventionnelle → salle de coronarographie

► NEUROLOGIQUE/AVC ischémique :
  - Position tête 0° (pas surélevée), glycémie contrôlée
  - Thrombolyse IV si <4h30 et pas de contre-indication (score NIHSS)
  - PAS d'antihypertenseurs sauf PA>220/120

- Voie, dose, débit : rigoureux. Jamais mélanger PO et IV sans préciser.
- Si urgence : "TRAITEMENT INITIAL — le traitement définitif sera adapté par l'équipe sur place"

══════════════════════════════════════════════════════════════
RÈGLE 3 — EXAMENS : UNIQUEMENT PERTINENTS POUR LA PATHOLOGIE
══════════════════════════════════════════════════════════════
Ne propose QUE les examens utiles pour confirmer ou traiter la pathologie dominante.
Pour chaque examen, indique brièvement son intérêt spécifique.

EXEMPLES :
► Pathologie RESPIRATOIRE : GDS artériels, Rx thorax, NFS-CRP, PCT si infection
  → PAS de BNP, PAS de troponine, PAS de coronarographie sauf signes cardiaques directs

► Pathologie CARDIAQUE : ECG, troponine, BNP, écho cardiaque
  → PAS de GDS en première intention sauf détresse respiratoire associée

► Sepsis : hémocultures x2, lactates, NFS-CRP-PCT, bilan rénal et hépatique

══════════════════════════════════════════════════════════════
RÈGLE 4 — ORIENTATION : SPÉCIALITÉ ADAPTÉE AU SYSTÈME
══════════════════════════════════════════════════════════════
Mapping pathologie → structure d'accueil correcte :
  BPCO/pneumonie/détresse respiratoire → Pneumologie / Réanimation respiratoire
  STEMI/insuffisance cardiaque aiguë   → Cardiologie interventionnelle / USIC
  AVC/TIA                              → Unité NeuroVasculaire (UNV)
  Sepsis/choc                          → Urgences + Réanimation polyvalente
  Appendicite/urgence chir             → Chirurgie digestive

⚠️ NE JAMAIS orienter vers cardiologie interventionnelle pour une pathologie respiratoire.
⚠️ NE JAMAIS orienter vers pneumologie pour un STEMI.

Toujours inclure : (1) numéro à appeler 15/18/112, (2) mode de transport SMUR si instable,
(3) type de structure précis (ex: "unité de soins intensifs respiratoires" pas "hôpital").

══════════════════════════════════════════════════════════════
RÈGLE 5 — RAISONNEMENT ÉTAPE PAR ÉTAPE (obligatoire)
══════════════════════════════════════════════════════════════
Dans "raisonnement_explicite", tu DOIS suivre cet ordre :
  Étape 1 : Quel système organe est principalement atteint et pourquoi ?
  Étape 2 : Quel diagnostic est le plus probable — quels signes le confirment ?
  Étape 3 : Quel diagnostic dangereux est écarté et pour quelle raison précise ?
  Étape 4 : Pourquoi ce traitement et pas un autre (justification spécifique au patient) ?

══════════════════════════════════════════════════════════════
RÈGLE 6 — HYPOTHÈSES
══════════════════════════════════════════════════════════════
- 1 hypothèse si diagnostic clairement dominant (>75%) et urgent à traiter
- 2 si deux diagnostics sérieusement à considérer (ex: BPCO vs OAP)
- 3 maximum. Ne force JAMAIS 3 si non justifié.
- Les facteurs de risque forts (ATCD familial précoce, tabagisme, hypercholestérolémie) priment sur l'âge.

══════════════════════════════════════════════════════════════
RÈGLE 7 — NIVEAUX DE CONDUITE
══════════════════════════════════════════════════════════════
- "maison"      : symptômes bénins, pas de signe d'alarme
- "laboratoire" : examens ambulatoires nécessaires, pas d'urgence vitale
- "hopital"     : urgence ou plateau technique requis — préciser le service EXACT

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


def _parse_diag(raw: str) -> "dict | None":
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

    # Contexte géographique
    patient_address = state.get("patient_address", "")
    geo_context = ""
    if patient_address:
        geo_context = f"""
CONTEXTE GÉOGRAPHIQUE ET ÉPIDÉMIOLOGIQUE :
Localisation du patient : {patient_address}

ADAPTATION GÉOGRAPHIQUE DU TRAITEMENT :
- Afrique subsaharienne : liste OMS essentiels, cotrimoxazole > amox-clav, artéméther-luméfantrine si paludisme probable, éviter chaîne froide difficile
- Maghreb / Afrique du Nord : génériques disponibles, automédication fréquente, coût à considérer
- Europe (FR/BE/CH) : protocoles HAS/NICE, ordonnance sécurisée, molécules remboursées
- Amérique latine / zones tropicales : toujours évoquer paludisme/dengue/typhoïde en DD si fièvre
"""
    else:
        geo_context = """
CONTEXTE : Protocoles standards européens / HAS / OMS si non précisé.
"""

    # Prompt partagé — identique pour A et B
    SHARED_PROMPT = f"""Prompt diagnostique optimisé :
{step3_output}

Données patient originales :
{ctx}
{geo_context}
INSTRUCTIONS CRITIQUES SUR LES HYPOTHÈSES :
- Toujours proposer PLUSIEURS hypothèses (idéalement 2-3) même si l'une domine
- Chaque hypothèse doit être cliniquement défendable, pas juste un remplissage
- Justifiez en citant les signes SPÉCIFIQUES qui penchent pour ou contre
- Si un diagnostic alternatif est sérieux, mettez-le même à 20-30%

INSTRUCTIONS CRITIQUES SUR LE TRAITEMENT (champ traitement_suggere) :
Tu es un clinicien senior. Pour le champ traitement_suggere, fournis un traitement CONCRET et IMMÉDIATEMENT ACTIONNABLE :

1. PRIORISE les traitements de la phase AIGUË (ce qu'on fait maintenant)
2. Format attendu — 3 à 5 lignes maximum, les plus importantes en premier :
   - Ligne 1 : traitement principal (molécule + voie + posologie + durée)
   - Ligne 2 : traitement complémentaire si nécessaire
   - Ligne 3 : alternative si allergie ou indisponibilité
   - Ligne 4 : mesure physique/non médicamenteuse clé (ex: oxygénothérapie si SpO2 < 94%)
   NE PAS inclure : vaccinations futures, conseils de prévention secondaire, suivi à long terme
3. Si affection VIRALE confirmée : état explicitement "Pas d'antibiotique indiqué (origine virale)"
4. Si HOSPITALISATION : citer les traitements IV/urgents à initier en arrivée
5. Si AUCUN traitement médicamenteux justifié : expliquer pourquoi en 1 phrase

Exemples de bonne formulation :
- "Amoxicilline 1g x3/j PO 7j. Alternative si allergie pénicilline: azithromycine 500mg J1 puis 250mg J2-J5. Paracétamol 1g x4/j si fièvre > 38.5°C."
- "Salbutamol nébulisé 2.5mg toutes 20min x3 (phase aiguë). Prednisolone 40mg/j PO 5j. Oxygénothérapie si SpO2 < 92%."
- "Pas d'antibiotique indiqué (rhinopharyngite virale). Paracétamol 1g x3/j. Sérum physiologique nasal 3x/j."

La conduite DOIT répondre aux 4 questions cliniques dans l'ordre de priorité :
Q1 — Quel diagnostic privilégier ? Q2 — Diagnostics redoutables à éliminer ?
Q3 — Examens complémentaires ? Q4 — Prise en charge initiale ?

⚠️ SI URGENCE VITALE (urgence="haute") :
- traitement_suggere : commencer par "⚠️ TRAITEMENT INITIAL (avant prise en charge définitive sur place) :"
  puis gestes dans l'ordre de priorité (voie veineuse → monitorage → molécule + voie + dose exacte)
  terminer par "Le traitement définitif sera adapté par l'équipe médicale sur place."
- instructions_immediates : commencer par "Appeler le 15 (SAMU) / 18 (pompiers) / 112 IMMÉDIATEMENT"
  préciser : ne pas se déplacer seul, attendre les secours ou ambulance/SMUR
- en_attendant_hopital : gestes précis, immédiats, réalisables sans médecin
  (position, O2 si disponible, voie veineuse, contre-indications immédiates, surveillance)
- raison_hospitalisation : niveau de soins requis + structure précise
  (ex: "Centre avec cardiologie interventionnelle - salle de coronarographie" vs "Urgences générales")

Génère le diagnostic. Réponds UNIQUEMENT en JSON valide :
{{
  "hypotheses": [
    {{"rang":1,"diagnostic":"...","probabilite":65,"cim10":"...","justification":"Si urgence vitale : commencer par 'URGENCE VITALE — [pattern reconnu]'. Puis : signes déterminants POUR et CONTRE.","examens":["..."]}},
    {{"rang":2,"diagnostic":"...","probabilite":25,"cim10":"...","justification":"Pourquoi ce diagnostic reste dans le différentiel","examens":["..."]}}
  ],
  "urgence":"haute|moyenne|faible",
  "score_confiance":85,
  "conduite":{{
    "niveau":"maison|laboratoire|hopital",
    "raison":"Q1 — [Diagnostic retenu] retenu car [signe clé]. Si urgence : 'URGENCE VITALE — action dans les 10 prochaines minutes.'",
    "diagnostics_a_eliminer":["Q2 — [Diagnostic grave 1] éliminé si [examen/critère]","[Diagnostic grave 2] : [comment l'exclure]"],
    "actes_labo":["Q3 — [Examen 1 : motif précis]","[Examen 2 pour éliminer diagnostic redoutable]"],
    "traitement_suggere":"Q4 — [Si urgence] ⚠️ TRAITEMENT INITIAL : 1. [Geste prioritaire + dose + voie]. 2. [Second geste]. 3. [Molécule SI stable : dose + voie]. Contre-indiqué si [contexte]. Traitement définitif sur place. [Si non-urgent] Molécule + posologie + durée + alternative si allergie.",
    "instructions_immediates":["[Si urgence] Appeler le 15 (SAMU) immédiatement — ne pas se déplacer seul","[Si urgence] Transport : ambulance/SMUR — pas de véhicule personnel","[Geste immédiat 2]"],
    "conseils_maison":["Mesure non médicamenteuse clé","Signe d'alarme → reconsulter immédiatement si...","Quand reconsulter (délai)"],
    "raison_hospitalisation":"[Service EXACT adapté au système atteint, ex: Pneumologie/Réa respiratoire pour pathologie pulmonaire, JAMAIS cardiologie si pas de signe cardiaque direct] — Appeler [15/18/112] — Transport : [SMUR si instable / ambulance / personnel si stable]",
    "en_attendant_hopital":["⏱️ EN ATTENDANT LES SECOURS — geste 1 précis avec technique (ex: position demi-assise si dyspnée)","Geste 2 (ex: O2 si disponible, objectif SpO2 > X%)","NE PAS donner : [contre-indications immédiates]","Surveiller : [constante critique] toutes les X minutes","Prévenir l'hôpital d'arrivée : [numéro]"]
  }},
  "raisonnement_explicite": "4 phrases obligatoires : (1) Système organe principal atteint : [respiratoire/cardiaque/neuro/etc] — preuves cliniques. (2) Diagnostic retenu : signe le plus déterminant POUR et pourquoi un autre a été écarté. (3) Traitement choisi : pourquoi indiqué pour CE patient, risque si mal appliqué, objectif thérapeutique précis. (4) Orientation : service EXACT choisi et pourquoi ce service et pas un autre."
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
    # Détection intelligente du type de fichier
    fname_lower = filename.lower()
    if media_type == "application/pdf":
        file_type = "document PDF médical"
    elif any(k in fname_lower for k in ["radio","rx","rayon","xray","scanner","irm","echo"]):
        file_type = "image radiologique"
    elif any(k in fname_lower for k in ["ecg","cardio","electro"]):
        file_type = "ECG ou tracé cardiaque"
    elif any(k in fname_lower for k in ["bilan","analyse","labo","resultat"]):
        file_type = "bilan biologique"
    elif media_type.startswith("image/"):
        file_type = "photo clinique"
    else:
        file_type = "document médical"

    prompt = f"""Analyse ce {file_type} nommé "{filename}" dans un contexte médical clinique.
IMPORTANT : Ne suppose pas le type de document, décris uniquement ce que tu vois réellement.
Fournis une description clinique courte (2-3 phrases max) :
- Décris précisément ce qui est visible (lésion, plaie, résultat, tracé, etc.)
- Mentionne les éléments anormaux ou cliniquement pertinents
- Évite les suppositions sur ce que c'est si ce n'est pas clair
- Si l'image est floue ou non médicale, indique-le simplement"""

    try:
        return call_vision(image_base64, media_type, prompt)
    except Exception as e:
        return f"[Analyse indisponible : {str(e)}]"

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