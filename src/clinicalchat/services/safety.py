"""Deterministic medicine safety (no AI): allergy, cross-reactivity and interaction checks from the knowledge graph."""

from __future__ import annotations

from typing import List, Optional, Sequence

from clinicalchat.domain.models import SafetyAlert
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.patients import PatientDirectory


def alert_line(a: SafetyAlert) -> str:
    tail = f" — {a.management}" if a.management else ""
    return f"[{a.level}] {a.drug} vs {a.against}: {a.message}{tail}"


class SafetyChecker:
    def __init__(self, kg: MedicalKnowledgeGraph, patients: PatientDirectory):
        self.kg = kg
        self.patients = patients

    def alerts(self, key: Optional[str], drugs: Sequence[str]) -> List[SafetyAlert]:
        found: List[SafetyAlert] = []
        for drug in drugs:
            found += self.kg.check_drug_for_patient(drug, key) if key else self.kg.check_drug(drug)
        uniq = {(a.level, a.category, a.drug, a.against, a.message): a for a in found}
        return sorted(uniq.values(), key=lambda a: a.rank)

    def contrast_alerts(self, key: str) -> List[str]:
        """Is iodinated contrast safe for this patient? (shown to the radiologist)"""
        return [alert_line(a) for a in self.alerts(key, ["iodinated contrast"])]

    def allergy_facts(self, key: Optional[str], drugs: Sequence[str], allergens: Sequence[str]) -> str:
        """Alerts, cross-reactivity and safe alternatives as text for the Insight agent."""
        lines = ["ALERT " + alert_line(a) for a in self.alerts(key, drugs)]
        p = self.patients.get(key)
        seen = set()
        for name in [*allergens, *[x.allergen for x in (p.allergies if p else [])]]:
            cross = self.kg.cross_reactive(name)
            sig = tuple(c["allergen"] for c in cross) or (name.lower(),)
            if sig in seen:
                continue
            seen.add(sig)
            if cross:
                lines.append(f"Cross-reactivity for {name}: " + "; ".join(f"{c['allergen']} ({c['risk']} risk: {c['reason']})" for c in cross))
            alts = self.kg.safe_alternatives(name)
            if alts:
                lines.append(f"Safe alternatives for {name}: " + ", ".join(alts))
        if drugs and not any(line.startswith("ALERT") for line in lines):
            lines.append(f"No allergy/interaction conflict found in the knowledge graph for: {', '.join(drugs)}"
                         + ("" if p else " (no patient selected — patient allergies not checked)"))
        return "\n".join(lines) or "None."
