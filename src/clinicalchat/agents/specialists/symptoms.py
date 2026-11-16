"""Symptom reasoning: which conditions could explain the symptoms (knowledge graph), and "Clinical Inconsistency"
when the symptoms are not explained by the reports or a report's own flag contradicts its reference range."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from clinicalchat.agents.specialists.lab import LabFindings
from clinicalchat.agents.specialists.radiology import ImagingFinding
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph


class SymptomReasoner:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg

    def differential(self, symptoms: Sequence[str], new_symptoms: Sequence[str],
                     patient_key: Optional[str]) -> List[Dict[str, object]]:
        """Conditions explaining the symptoms reported right now come first, then the full symptom history."""
        if not symptoms:
            return []
        merged: Dict[str, Dict[str, object]] = {}
        recent = self.kg.differential(new_symptoms, patient_key) if new_symptoms else []
        for d in [*recent, *self.kg.differential(symptoms, patient_key)]:
            merged.setdefault(str(d["condition"]), d)
        return list(merged.values())

    def inconsistencies(self, new_symptoms: Sequence[str], differential: Sequence[Dict[str, object]], labs: LabFindings,
                        imaging: Iterable[ImagingFinding], documented: Sequence[str]) -> List[str]:
        imaging = list(imaging)
        issues: List[str] = []
        if new_symptoms and differential and (labs.results or imaging):
            candidates = [str(d["condition"]) for d in differential]
            supported = set(documented) | {c["condition"] for c in labs.conditions}
            for f in imaging:
                supported.update(self.kg.conditions_in_text(f.section.text))
            if not any(c in supported for c in candidates):
                issues.append(f"Reported symptoms ({', '.join(new_symptoms)}) are not explained by the available reports "
                              f"(symptom-based candidates: {', '.join(candidates[:4])}) — further evaluation needed.")
        for r, src in labs.results:
            if r.status_discrepancy:
                issues.append(f"{src}: report flags {r.parameter} as '{r.reported_status}' but {r.value_text} is "
                              f"{r.status.value} against the printed range {r.reference_range}.")
        return issues
