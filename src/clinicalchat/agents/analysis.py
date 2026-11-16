"""🔬 Clinical Analysis Agent (deterministic, no LLM).

Runs the specialist analysers from the problem statement on the records the Extraction agent selected:
Lab Interpreter, Radiology Analyzer, Allergy Safety, plus symptom reasoning (differential + consistency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from clinicalchat.agents.extraction import ExtractionResult
from clinicalchat.agents.specialists.allergy import AllergyFindings, AllergySafety
from clinicalchat.agents.specialists.lab import LabFindings, interpret_labs
from clinicalchat.agents.specialists.radiology import ImagingFinding, analyse_imaging
from clinicalchat.agents.specialists.symptoms import SymptomReasoner
from clinicalchat.domain.models import Domain, LabResult, SafetyAlert
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.catalog import DocumentCatalog
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.safety import SafetyChecker


@dataclass
class AnalysisResult:
    lab_results: List[tuple[LabResult, str]] = field(default_factory=list)  # (result, source)
    abnormal: List[tuple[LabResult, str]] = field(default_factory=list)
    lab_conditions: List[Dict[str, str]] = field(default_factory=list)  # {condition, evidence}
    imaging: List[ImagingFinding] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    drug_alerts: List[SafetyAlert] = field(default_factory=list)  # drugs asked about
    medication_alerts: List[SafetyAlert] = field(default_factory=list)  # current medicines
    allergy_facts: str = ""
    differential: List[Dict[str, object]] = field(default_factory=list)
    documented_conditions: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    specialists: List[str] = field(default_factory=list)  # which analysers had data

    @property
    def possible_conditions(self) -> List[str]:
        names = [*self.documented_conditions, *(c["condition"] for c in self.lab_conditions)]
        names += [str(d["condition"]) for d in self.differential[:3]]
        return list(dict.fromkeys(names))

    def facts_text(self) -> str:
        """Deterministic findings handed to the Insight agent (it must not contradict them)."""
        out = self._lab_facts() + self._imaging_facts() + self._allergy_facts()
        if self.differential:
            out.append("SYMPTOM DIFFERENTIAL (knowledge graph): " + "; ".join(
                f"{d['condition']} (score {d['score']}; confirm with {', '.join(d.get('workup', [])[:3]) or 'clinical review'})"
                for d in self.differential[:5]))
        if self.documented_conditions:
            out.append("DOCUMENTED CONDITIONS: " + ", ".join(self.documented_conditions))
        out += [f"CLINICAL INCONSISTENCY: {i}" for i in self.inconsistencies]
        return "\n".join(out) or "No structured findings."

    def _lab_facts(self) -> List[str]:
        if not self.lab_results:
            return []
        return ([f"LAB INTERPRETER: {len(self.lab_results)} values, {len(self.abnormal)} outside reference range."]
                + [f"- {r.line()} [{src}]" for r, src in self.abnormal[:20]]
                + [f"- Pattern suggests {c['condition']} ({c['evidence']})" for c in self.lab_conditions[:6]])

    def _imaging_facts(self) -> List[str]:
        out = []
        for f in self.imaging:
            out.append(f"RADIOLOGY ANALYZER — {f.section.title} [{f.source}]: impression: {f.impression or 'not stated'}")
            out += [f"- RED FLAG: {h}" for h in f.red_flags[:3]] + [f"- abnormal: {h}" for h in f.abnormal[:3]]
        return out

    def _allergy_facts(self) -> List[str]:
        if not (self.allergies or self.drug_alerts or self.medication_alerts):
            return []
        out = ["ALLERGY SAFETY: documented allergies: " + (", ".join(self.allergies) or "none recorded")]
        out += [f"- ALERT [{a.level}] {a.drug} vs {a.against}: {a.message}" for a in self.drug_alerts]
        out += [f"- current-medication alert [{a.level}] {a.drug} vs {a.against}: {a.message}" for a in self.medication_alerts[:5]]
        if self.allergy_facts and self.allergy_facts != "None.":
            out.append(self.allergy_facts)
        return out


class ClinicalAnalysisAgent:
    def __init__(self, catalog: DocumentCatalog, patients: PatientDirectory, kg: MedicalKnowledgeGraph,
                 safety: SafetyChecker):
        self.catalog = catalog
        self.patients = patients
        self.kg = kg
        self.allergy = AllergySafety(kg, safety)
        self.symptoms = SymptomReasoner(kg)

    def run(self, ext: ExtractionResult, drugs: Sequence[str], allergens: Sequence[str],
            check_allergies: bool = True) -> AnalysisResult:
        key = ext.patient_key
        res = AnalysisResult()
        labs = self._labs(ext)
        res.lab_results, res.abnormal, res.lab_conditions = labs.results, labs.abnormal, labs.conditions
        res.imaging = [analyse_imaging(s, self.catalog.source_of(s)) for s in ext.sections
                       if self.catalog.domain_of(s) == Domain.RADIOLOGY]
        if res.lab_results:
            res.specialists.append("Lab Interpreter")
        if res.imaging:
            res.specialists.append("Radiology Analyzer")
        if check_allergies:
            self._apply_allergy(res, self.allergy.check(self.patients.get(key), drugs, allergens), drugs, allergens)
        res.differential = self.symptoms.differential(ext.symptoms, ext.new_symptoms, key)
        if key:
            res.documented_conditions = [c["condition"] for c in self.kg.patient_conditions(key) if c["basis"] == "documented"]
        res.inconsistencies = self.symptoms.inconsistencies(ext.new_symptoms, res.differential, labs, res.imaging,
                                                            res.documented_conditions)
        return res

    def _labs(self, ext: ExtractionResult) -> LabFindings:
        return interpret_labs((r, self.catalog.source_of(s)) for s in ext.sections
                              if self.catalog.domain_of(s) == Domain.LAB or s.lab_results for r in s.lab_results)

    @staticmethod
    def _apply_allergy(res: AnalysisResult, found: AllergyFindings, drugs: Sequence[str], allergens: Sequence[str]) -> None:
        res.allergies, res.drug_alerts = found.allergies, found.drug_alerts
        res.medication_alerts, res.allergy_facts = found.medication_alerts, found.facts
        if found.has_data(drugs, allergens):
            res.specialists.append("Allergy Safety")
