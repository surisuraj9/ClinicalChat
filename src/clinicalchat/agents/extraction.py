"""📄 Data Extraction Agent (deterministic, no LLM).

Selects the patient's records that matter for a question (respecting the role's data access), scores how reliably
they were read and lists missing / ambiguous information ("is there enough data to analyse?").
Reading uploaded files is done in ``ingestion/uploads.py``; this agent works on what is already on record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from clinicalchat.domain.models import Domain, Section
from clinicalchat.domain.roles import Role, RolePolicy, policy
from clinicalchat.ingestion.quality import ExtractionQuality, assess_section, summarise_quality
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.catalog import DocumentCatalog
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.records import RecordRetriever

# question keyword -> words that identify the matching lab section title
LAB_PANELS = {
    r"\bcbc\b|blood count|haemoglobin|hemoglobin|\bwbc\b|platelet|anaemi|anemi": ["cbc"],
    r"thyroid|\btsh\b|\bt3\b|\bt4\b": ["thyroid"],
    r"liver|\blft\b|bilirubin|\balt\b|\bast\b|sgpt|sgot": ["liver"],
    r"kidney|renal|\bkft\b|\brft\b|creatinine|urea|\begfr\b": ["kidney"],
    r"sugar|glucose|diabet|hba1c": ["sugar"],
    r"heart|cardiac|lipid|cholesterol|triglycer|troponin": ["heart"],
    r"\bbmi\b|body mass|weight": ["bmi"],
    r"metabolic|\bcmp\b|\bbmp\b|electrolyte": ["liver", "kidney", "sugar"],
}
IMAGING_TERMS = {
    "ct": r"\bct\b|computed tomograph", "mri": r"\bmri\b|magnetic resonance", "x-ray": r"x-?ray|\bcxr\b|radiograph",
    "ultrasound": r"ultrasound|\busg\b|sonograph|doppler", "chest": r"chest|lung|thora", "brain": r"brain|head|cranial",
    "abdomen": r"abdom|liver|appendi|gall|pelvi", "spine": r"spine|spinal|lumbar|cervical|disc",
    "mammogram": r"mammo|breast",
}


@dataclass
class ExtractionResult:
    patient_key: Optional[str]
    focus: List[Domain] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)  # canonical symptoms on record (incl. this message)
    new_symptoms: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    denied: List[Domain] = field(default_factory=list)
    quality: ExtractionQuality = field(default_factory=ExtractionQuality)
    sufficient: bool = True
    reference_only: bool = False


class DataExtractionAgent:
    def __init__(self, catalog: DocumentCatalog, patients: PatientDirectory, kg: MedicalKnowledgeGraph,
                 records: RecordRetriever):
        self.catalog = catalog
        self.patients = patients
        self.kg = kg
        self.records = records

    def run(self, question: str, role: Role, patient_key: Optional[str], focus: Sequence[Domain],
            new_symptoms: Sequence[str] = (), upload_doc_ids: Sequence[str] = ()) -> ExtractionResult:
        pol = policy(role)
        res = ExtractionResult(patient_key=patient_key, new_symptoms=list(new_symptoms))
        res.denied = [d for d in focus if d not in pol.domains]
        res.focus = [d for d in focus if d in pol.domains]
        if patient_key:
            self._patient_records(res, question, pol, upload_doc_ids)
        else:
            self._reference_records(res, question, pol)
        patient = self.patients.get(patient_key)
        qualities = [assess_section(s, self.catalog.document_of(s), patient.name if patient else None)
                     for s in res.sections if self.catalog.domain_of(s) != Domain.REFERENCE]
        res.quality = summarise_quality(qualities)
        res.sources = list(dict.fromkeys(self.catalog.source_of(s) for s in res.sections))
        res.sufficient = bool(res.sections) or bool(res.symptoms)
        return res

    def _patient_records(self, res: ExtractionResult, question: str, pol: RolePolicy,
                         upload_doc_ids: Sequence[str]) -> None:
        key = res.patient_key
        visible = [d for d in Domain if d in pol.domains and d != Domain.REFERENCE]
        sections = self.catalog.patient_sections(key, res.focus or visible)
        if upload_doc_ids:
            sections = [s for s in sections if s.doc_id in set(upload_doc_ids)] or sections
        else:
            sections = self._focus(question, sections)
        res.sections = sections
        res.symptoms = list(dict.fromkeys([*self.kg.patient_symptoms(key), *res.new_symptoms]))
        on_file = set(self.patients.domains_of(key))
        res.missing += [f"No {d.value.replace('_', ' ')} reports on file" for d in res.focus
                        if d != Domain.REFERENCE and d not in on_file]
        if not sections and not res.missing:
            res.missing.append("No diagnostic reports on file — upload any reports you have (📎)")
        patient = self.patients.get(key)
        if patient and patient.age is None:
            res.missing.append("Age not recorded (age-specific thresholds cannot be applied)")

    def _reference_records(self, res: ExtractionResult, question: str, pol: RolePolicy) -> None:
        domains = [*res.focus, Domain.REFERENCE] if res.focus else None
        res.sections = [s for s in self.records.search_sections(question, domains=domains)
                        if self.catalog.domain_of(s) in pol.domains]
        res.reference_only = True
        res.symptoms = list(res.new_symptoms)

    def _focus(self, question: str, sections: List[Section]) -> List[Section]:
        """"Explain my thyroid results" -> only the thyroid panel; "my chest x-ray" -> only matching imaging."""
        q = question.lower()
        panels = {w for rx, words in LAB_PANELS.items() if re.search(rx, q) for w in words}
        modal = [t for t, rx in IMAGING_TERMS.items() if re.search(rx, q)]
        if not panels and not modal:
            return sections
        keep = [s for s in sections if self._matches(s, panels, modal)]
        return keep or sections

    def _matches(self, s: Section, panels: set, modal: List[str]) -> bool:
        if s.lab_results:
            head = (s.title + " " + self.catalog.source_of(s)).lower()
            return bool(panels) and any(w in head for w in panels)
        text = (s.title + " " + s.text[:300]).lower()
        return bool(modal) and any(re.search(IMAGING_TERMS[t], text) for t in modal)
