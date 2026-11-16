"""Allergy Safety: documented allergies, drugs asked about, current medicines (knowledge-graph checks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from clinicalchat.domain.models import PatientProfile, SafetyAlert
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.safety import SafetyChecker


@dataclass
class AllergyFindings:
    allergies: List[str] = field(default_factory=list)
    drug_alerts: List[SafetyAlert] = field(default_factory=list)  # drugs asked about
    medication_alerts: List[SafetyAlert] = field(default_factory=list)  # current medicines
    facts: str = ""

    def has_data(self, drugs: Sequence[str], allergens: Sequence[str]) -> bool:
        return bool(self.allergies or self.drug_alerts or drugs or allergens)


class AllergySafety:
    def __init__(self, kg: MedicalKnowledgeGraph, safety: SafetyChecker):
        self.kg = kg
        self.safety = safety

    def check(self, patient: Optional[PatientProfile], drugs: Sequence[str], allergens: Sequence[str]) -> AllergyFindings:
        key = patient.patient_key if patient else None
        out = AllergyFindings()
        if patient:
            out.allergies = [f"{a.allergen} ({a.severity.lower()})" for a in patient.allergies]
            out.medication_alerts = self.kg.medication_review(key)
        out.drug_alerts = self.safety.alerts(key, drugs)
        if drugs or allergens or (patient and patient.allergies):
            out.facts = self.safety.allergy_facts(key, drugs, allergens)
        return out
