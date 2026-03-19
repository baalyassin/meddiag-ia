[README.md](https://github.com/user-attachments/files/26121621/README.md)
# MedDiag IA 🏥

> **Diagnostic médical assisté par intelligence artificielle**  
> Pipeline multi-LLM pour les déserts médicaux — infirmier(e)s & médecins

---

## Vue d'ensemble

MedDiag IA est une plateforme web de diagnostic médical assisté par IA, conçue pour les zones sous-dotées en médecins. Elle structure le travail de l'infirmier(e) de terrain, génère des hypothèses diagnostiques via 3 modèles LLM en parallèle, adapte les traitements au contexte géographique et soumet le tout à la validation obligatoire du médecin.

**L'IA propose — le médecin décide. Toujours.**

---

## Fonctionnalités

- 🎙️ **Collecte multimodale** — saisie texte, dictée vocale, consultation vidéo, import documents/photos médicales
- ⚡ **Auto-extraction des constantes** — TA, FC, température, SpO2, glycémie, IMC depuis le texte libre
- 🧠 **Pipeline multi-LLM** — 3 modèles en parallèle (llama-3.3, kimi-k2) + arbitrage (gpt-oss-120b)
- 🌍 **Traitement géolocalisé** — médicaments adaptés au contexte (OMS, HAS, protocoles locaux)
- 🚨 **Détection d'urgences vitales** — STEMI, AVC, sepsis, détresse respiratoire → actions immédiates priorisées
- 👩‍⚕️ **Interface duale** — workflow infirmier / médecin séparé avec signature électronique
- 📄 **Export PDF officiel** — rapport médical avec tampon, CIM-10, signature

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│              Next.js 16 / React                         │
│         Interface Infirmier  |  Interface Médecin        │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────┐
│                    BACKEND                              │
│              FastAPI (Python 3.12)                      │
│              Validation Pydantic                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              PIPELINE LANGGRAPH (4 nœuds)               │
│                                                         │
│  ① Initialisation  →  JSON normalisé                    │
│  ② Génération      →  contexte épidémiologique          │
│  ③ Évaluation      →  optimisation du prompt            │
│  ④ Prédictif       →  IA A + IA B ──► Arbitre C         │
│                        llama-3.3   kimi-k2   gpt-oss-120b│
└─────────────────────────────────────────────────────────┘
```

---

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Frontend | Next.js 16, React, Tailwind |
| Backend | FastAPI, Python 3.12, Uvicorn |
| Orchestration IA | LangGraph |
| LLM — Structuration | `meta-llama/llama-4-scout-17b` (Groq) |
| LLM — IA A | `llama-3.3-70b-versatile` (Groq) |
| LLM — IA B | `moonshotai/kimi-k2-instruct` (Groq) |
| LLM — Arbitre | `openai/gpt-oss-120b` (Groq) |
| Vision | `llama-4-scout-17b` — analyse ECG, photos |
| Géolocalisation | OpenStreetMap Nominatim + Google Maps |

---

## Installation

### Prérequis

- Python 3.12+
- Node.js 18+
- Clé API Groq

### Backend

```bash
cd /chemin/vers/pipelineoff
pip install -r requirements.txt

# Créer le fichier .env
echo "GROQ_API_KEY=gsk_votre_clé_ici" > .env

# Lancer le serveur
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

L'application est accessible sur `http://localhost:3000`

---

## Structure du projet

```
pipelineoff/
├── main.py                  # FastAPI — endpoints REST
├── pipeline/
│   └── nodes.py             # LangGraph — 4 nœuds diagnostiques
├── models/
│   └── schemas.py           # Pydantic — modèles de données
├── frontend/
│   └── app/
│       ├── page.js
│       └── components/
│           ├── MedicalPipeline.jsx   # Interface principale
│           └── WebcamCapture.jsx     # Module vidéo
└── .env                     # GROQ_API_KEY (non commité)
```

---

## Variables d'environnement

```env
GROQ_API_KEY=gsk_...
```

---

## Workflow clinique

```
Infirmier(e)                    Système IA                    Médecin
    │                               │                             │
    ├── Saisit symptômes ─────────► │                             │
    ├── Dicte l'anamnèse            │                             │
    ├── Importe documents           │ ① Structuration             │
    ├── Renseigne constantes        │ ② Contexte géo              │
    │                               │ ③ Optimisation prompt       │
    │                               │ ④ IA A + IA B               │
    │                               │      └► Arbitre C           │
    │ ◄─── Diagnostic + conduite ── │                             │
    │                               │                             │
    ├── Consulte les résultats      │                             │
    ├── Transmet au médecin ─────────────────────────────────────►│
    │                               │                             │
    │                               │                ├── Valide / corrige
    │                               │                ├── Prescrit
    │                               │                └── Signe
    │ ◄─── Plan de soins ───────────────────────────────────────  │
```

---

## API — Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/extract` | Structuration des données patient brutes |
| `POST` | `/api/pipeline/run` | Lancement du pipeline diagnostique complet |
| `POST` | `/api/analyze-document` | Analyse vision d'un document médical ou ECG |
| `POST` | `/api/questions/generate` | Génération de questions diagnostiques dynamiques |
| `GET`  | `/api/export/prompt` | Export du prompt final envoyé aux modèles |

---

## Rôles utilisateurs

### Infirmier(e)
- Collecte des données patient (phase 1)
- Vérification et complétion du dossier structuré (phase 2)
- Consultation du diagnostic IA (phase 4)
- Transmission du dossier au médecin

### Médecin
- Accès à l'interface médecin dédiée
- Consultation du consensus multi-modèles
- Validation ou correction du diagnostic
- Prescription et plan de traitement
- Signature électronique + export PDF

---

## Sécurité et conformité

- 🔒 Chiffrement TLS 1.3 sur toutes les communications
- 🏥 Architecture compatible hébergement HDS
- 🇪🇺 Conception Privacy by Design (RGPD Art. 9)
- 📋 Journaux d'audit de chaque accès patient
- 🔐 Pseudonymisation avant envoi aux API LLM tierces

---

## Limites importantes

> ⚠️ **MedDiag IA est un outil d'aide à la décision médicale, pas un outil de diagnostic autonome.**
> - Toute décision clinique doit être validée par un médecin habilité
> - Le système n'effectue aucune prescription directe
> - Les hypothèses diagnostiques sont indicatives et non définitives
> - La responsabilité médicale reste intégralement chez le médecin signataire

---

## Contribuer

```bash
git checkout -b feature/ma-fonctionnalite
git commit -m "feat: description"
git push origin feature/ma-fonctionnalite
```

---

## Licence

Projet privé — © 2025 MedDiag IA. Tous droits réservés.

---

*Fait avec ❤️ pour les déserts médicaux*
