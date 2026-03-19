"""

PROTECTION – DROITS D’AUTEUR – 2026

 

© 2026 Prénom NOM1, Prénom NOM2, Prénom NOM3

 

Ce travail (code source + documents associés) est protégé par le droit d’auteur.

Autorisation limitée à la lecture seule pour évaluation du cours uniquement.

Aucune cession de droits. Toute autre utilisation (reproduction, modification,

exploitation pédagogique ou commerciale) interdite sans accord écrit préalable.

"""



"""
Schémas Pydantic — Pipeline Diagnostic Médical v2
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class UrgenceLevel(str, Enum):
    haute   = "haute"
    moyenne = "moyenne"
    faible  = "faible"


class VitalSigns(BaseModel):
    tension:     Optional[str] = None
    fc:          Optional[str] = None
    temperature: Optional[str] = None
    poids:       Optional[str] = None
    spo2:        Optional[str] = None
    glycemie:    Optional[str] = None

    @field_validator("tension","fc","temperature","poids","spo2","glycemie", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if v is not None else v


class PatientRawData(BaseModel):
    transcription:      str
    vital_signs:        VitalSigns = Field(default_factory=VitalSigns)
    confirmed_symptoms: list[str]  = Field(default_factory=list)
    images_base64:      list[str]  = Field(default_factory=list)


class StructuredPatientData(BaseModel):
    motif_consultation:  str
    symptomes_principaux: list[str]        = Field(default_factory=list)
    constantes_vitales:  VitalSigns        = Field(default_factory=VitalSigns)
    antecedents:         Optional[str]     = None
    symptomes_confirmes: list[str]         = Field(default_factory=list)
    urgence_estimee:     Optional[UrgenceLevel] = None
    contexte_clinique:   Optional[str]     = None
    score_gravite:       Optional[int]     = Field(None, ge=0, le=10)


class PipelineRequest(BaseModel):
    structured_data:  StructuredPatientData
    image_analysis:   Optional[str] = None   # résultat analyse vision injecté dans nœud ①
    patient_address:  Optional[str] = None   # adresse patient pour contexte épidémiologique


class DiagnosticHypothesis(BaseModel):
    rang:          int
    diagnostic:    str
    probabilite:   int
    cim10:         Optional[str] = None
    justification: str
    examens:       list[str] = Field(default_factory=list)


class ConduiteASuivre(BaseModel):
    niveau:                  str            # "maison" | "laboratoire" | "hopital"
    raison:                  str
    instructions_immediates: list[str]      = Field(default_factory=list)
    conseils_maison:         list[str]      = Field(default_factory=list)
    actes_labo:              list[str]      = Field(default_factory=list)
    diagnostics_a_eliminer:  list[str]      = Field(default_factory=list)
    traitement_suggere:      Optional[str]  = None
    raison_hospitalisation:  Optional[str]  = None
    en_attendant_hopital:    list[str]      = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    hypotheses:      list[DiagnosticHypothesis]
    urgence:         UrgenceLevel
    score_confiance: int
    conduite:        ConduiteASuivre


class PipelineResult(BaseModel):
    step_outputs: dict[str, str]   = Field(default_factory=dict)
    diagnostic:   DiagnosticResult