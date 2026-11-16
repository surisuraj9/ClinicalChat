"""Patient directory: profiles (dataset + self-registered), name lookup and the summary handed to the AI."""

from __future__ import annotations

import re
import threading
from typing import List, Optional

from clinicalchat.domain.models import AllergyRecord, Domain, PatientProfile
from clinicalchat.ingestion.registry import PatientRegistry, patient_key
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.catalog import DocumentCatalog

SELF_REPORTED = "Self-reported by patient"


class PatientDirectory:
    def __init__(self, registry: PatientRegistry, kg: MedicalKnowledgeGraph, catalog: DocumentCatalog,
                 lock: threading.RLock):
        self.registry = registry
        self.kg = kg
        self.catalog = catalog
        self._lock = lock

    def get(self, key: Optional[str]) -> Optional[PatientProfile]:
        return self.registry.get(key) if key else None

    def all(self) -> List[PatientProfile]:
        return self.registry.all()

    def names(self) -> List[str]:
        return [p.name for p in self.registry.patients.values()]

    def find_in_text(self, text: str) -> Optional[str]:
        """Patient named in the question (full name); prefers the record with the most data when names repeat."""
        matches = self.registry.search(text)
        if not matches:
            return None
        age = re.search(r"\b(\d{1,3})\s*(?:y|yr|yrs|years?)\b", text.lower())
        if age:
            matches = [p for p in matches if p.age == int(age.group(1))] or matches
        return max(matches, key=lambda p: (len(p.name), len(p.section_ids))).patient_key

    def domains_of(self, key: str) -> List[Domain]:
        found = {self.catalog.domain_of(s) for s in self.catalog.patient_sections(key)} - {None}
        return sorted(found, key=lambda d: d.value)

    def allergy_text(self, key: Optional[str]) -> str:
        p = self.get(key)
        return ", ".join(sorted(a.allergen for a in p.allergies)) if p and p.allergies else "none recorded"

    def summary(self, key: Optional[str]) -> str:
        p = self.get(key)
        if not p:
            return "No specific patient selected."
        lines = [f"Name: {p.name}", f"Age: {p.age if p.age is not None else 'unknown'}", f"Gender: {p.gender or 'unknown'}"]
        lines.append("Documented allergies: " + (", ".join(f"{a.allergen} ({a.severity})" for a in p.allergies) or "none recorded"))
        if p.medications:
            lines.append("Medications: " + ", ".join(p.medications))
        conds = [c["condition"] for c in self.kg.patient_conditions(key)]
        if conds or p.conditions:
            lines.append("Conditions: " + ", ".join(dict.fromkeys([*p.conditions, *conds])))
        if p.symptoms:
            lines.append("Reported symptoms: " + "; ".join(p.symptoms[-5:]))
        lines.append("Records: " + (", ".join(p.sources) or "none on file yet"))
        return "\n".join(lines)

    # ------------------------------------------------------------------ changes
    def new_key(self, name: str, age: Optional[int]) -> str:
        """Never merge a self-registered user into an existing record with the same name (privacy)."""
        base = patient_key(name, age)
        key, n = base, 1
        while key in self.registry.patients:
            n += 1
            key = f"{base}-{n}"
        return key

    def create(self, key: str, name: str, age: Optional[int], gender: Optional[str]) -> PatientProfile:
        with self._lock:
            profile = self.registry.patients.get(key) or PatientProfile(patient_key=key, name=name, age=age, gender=gender)
            self.registry.patients[key] = profile
            self.kg.refresh_patient(profile)
            return profile

    def apply_fact(self, key: str, kind: str, value: str, refresh: bool = True) -> None:
        """Add a symptom / allergy / medicine to the in-memory profile and the knowledge graph."""
        p = self.get(key)
        if not p or not value.strip():
            return
        with self._lock:
            if kind == "symptom" and value not in p.symptoms:
                p.symptoms.append(value)
            elif kind == "allergy" and value.lower() not in {a.allergen.lower() for a in p.allergies}:
                allergen = self.kg.resolve_allergen(value) or value.strip()
                p.allergies.append(AllergyRecord(allergen=allergen, category="self-reported", severity="UNKNOWN",
                                                 source=SELF_REPORTED, evidence=[SELF_REPORTED]))
            elif kind == "medication" and value not in p.medications:
                p.medications.append(value)
            if refresh:
                self.kg.refresh_patient(p)

    def refresh(self, key: str) -> None:
        self.kg.refresh_patient(self.registry.patients[key])
