"""📄 Data Extraction Agent (deterministic, no LLM).

* Uploads: parse a report (DOCX/PDF/TXT/XLSX/image with OCR), classify it, map it to the patient and score how
  reliably it was structured. Low-confidence extractions are flagged for clinician review.
* Questions: select the patient's records that are relevant (respecting the role's data access), list the
  structured values found and detect missing / ambiguous information ("is there enough data to analyse?").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel, Field

from clinicalchat.dates import date_from_fields, date_from_text
from clinicalchat.ingestion.lab_tables import make_lab_result
from clinicalchat.ingestion.loaders import load_document
from clinicalchat.ingestion.sectioning import split_sections
from clinicalchat.models import ClinicalDocument, Domain, LabResult, LabStatus, Section
from clinicalchat.rbac import Role, policy

if TYPE_CHECKING:
    from clinicalchat.engine import ClinicalEngine

CONFIDENCE_THRESHOLD = 0.55
MAX_UPLOAD_FILES = 10
SUPPORTED_UPLOADS = [".pdf", ".docx", ".txt", ".md", ".xlsx", ".png", ".jpg", ".jpeg"]

REPORT_TYPES = {
    "lab": ("lab", Domain.LAB),
    "radiology": ("radiology", Domain.RADIOLOGY),
    "allergy": ("allergy_upload", Domain.ALLERGY),
    "clinical_note": ("patient_record", Domain.PATIENT_RECORD),
}
_RAD_WORDS = re.compile(r"\b(x-?ray|radiograph|ct\b|computed tomography|mri|ultrasound|usg|sonograph|mammogra|impression|findings)\b", re.I)
_RAD_STUDY = re.compile(r"\b(x-?ray|radiograph|ct\b|computed tomography|mri|ultrasound|usg|sonograph|mammogra|scan)\b", re.I)
_LAB_WORDS = re.compile(r"\b(haemoglobin|hemoglobin|cbc|creatinine|egfr|tsh|hba1c|glucose|cholesterol|reference range|platelet|bilirubin)\b", re.I)
_ALLERGY_WORDS = re.compile(r"\b(allerg\w*|ige|skin prick|anaphyla\w*|urticaria)\b", re.I)
_PRINTED_NAME = re.compile(r"^(?:Patient[ \t]*Name|Name|Patient)[ \t]*[:\-][ \t]*([A-Z][a-zA-Z.'’]+(?:[ \t]+[A-Z][a-zA-Z.'’]+){0,3})", re.M)

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


# ---------------------------------------------------------------------- quality scoring
class SectionQuality(BaseModel):
    section_id: str
    title: str
    source: str = ""
    report_type: str = "clinical_note"
    report_date: Optional[str] = None
    score: float = 1.0
    lab_values: int = 0
    abnormal_values: int = 0
    issues: List[str] = Field(default_factory=list)


class ExtractionQuality(BaseModel):
    score: float = 1.0
    level: str = "HIGH"  # HIGH | MEDIUM | LOW
    low_confidence: bool = False
    method: str = "text"
    sections: List[SectionQuality] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


def classify(text: str, lab_count: int, allergy_panel: bool = False) -> tuple[str, Domain]:
    """`allergy_panel`: every parsed value is an IgE measurement (allergy test reports are table-shaped too)."""
    lab_hits = len(_LAB_WORDS.findall(text))
    if allergy_panel and len(_ALLERGY_WORDS.findall(text)) >= 2:
        return "allergy_upload", Domain.ALLERGY
    if lab_count >= 3 or lab_hits >= 3:
        return "lab", Domain.LAB
    if _RAD_STUDY.search(text) and len(_RAD_WORDS.findall(text)) >= 2:
        return "radiology", Domain.RADIOLOGY
    if len(_ALLERGY_WORDS.findall(text)) >= 2:
        return "allergy_upload", Domain.ALLERGY
    if lab_count:
        return "lab", Domain.LAB
    return "patient_record", Domain.PATIENT_RECORD


def _report_type(doc: Optional[ClinicalDocument]) -> str:
    if doc is None:
        return "clinical_note"
    return {Domain.LAB: "lab", Domain.RADIOLOGY: "radiology", Domain.ALLERGY: "allergy",
            Domain.PATIENT_RECORD: "clinical_note", Domain.REFERENCE: "reference"}[doc.domain]


def printed_name(text: str) -> Optional[str]:
    m = _PRINTED_NAME.search(text)
    return m.group(1).strip() if m else None


def names_match(printed: str, expected: str) -> bool:
    a, b = printed.lower().strip(), expected.lower().strip()
    return a in b or b in a


def foreign_name(doc: ClinicalDocument, expected_name: str) -> Optional[str]:
    """Name printed on an uploaded report when it clearly belongs to someone else."""
    for s in doc.sections:
        printed = printed_name(s.text)
        if printed and expected_name and not names_match(printed, expected_name):
            return printed
    return None


def assess_section(section: Section, doc: Optional[ClinicalDocument], expected_name: Optional[str] = None) -> SectionQuality:
    rtype = _report_type(doc)
    q = SectionQuality(section_id=section.section_id, title=section.title, source=doc.filename if doc else "",
                       report_type=rtype, report_date=section.report_date, lab_values=len(section.lab_results),
                       abnormal_values=sum(1 for r in section.lab_results if r.abnormal))
    penalties: List[tuple[float, str]] = []
    text_len = len(section.text.strip())
    if text_len < 150:
        penalties.append((0.35, f"very little text extracted ({text_len} characters)"))
    if doc is not None and doc.extraction_method == "ocr":
        penalties.append((0.15, "text read by OCR from a scanned image — verify values against the original"))
    if not section.report_date:
        penalties.append((0.05, "no report/test date found"))
    if rtype == "lab":
        labs = section.lab_results
        if not labs:
            penalties.append((0.35, "lab report but no structured values could be parsed"))
        else:
            with_range = sum(1 for r in labs if r.ref_low is not None or r.ref_high is not None)
            if with_range / len(labs) < 0.7:
                penalties.append((0.2, f"reference ranges missing for {len(labs) - with_range} of {len(labs)} values"))
            unknown = sum(1 for r in labs if r.status == LabStatus.UNKNOWN)
            if unknown / len(labs) > 0.3:
                penalties.append((0.1, f"{unknown} values could not be classified against a range"))
            disc = [r.parameter for r in labs if r.status_discrepancy]
            if disc:
                penalties.append((min(0.15, 0.05 * len(disc)), "report flag disagrees with printed range: " + ", ".join(disc[:5])))
    elif rtype == "radiology":
        low = section.text.lower()
        if "impression" not in low and "conclusion" not in low:
            penalties.append((0.15, "no impression/conclusion section found"))
        if "finding" not in low and "observation" not in low:
            penalties.append((0.1, "no findings section found"))
    printed = printed_name(section.text)
    if expected_name and printed and not names_match(printed, expected_name):
        penalties.append((0.5, f"name on report ('{printed}') does not match the patient ('{expected_name}')"))
    q.score = round(max(0.0, 1.0 - sum(p for p, _ in penalties)), 2)
    q.issues = [msg for _, msg in penalties]
    return q


def summarise_quality(qualities: Sequence[SectionQuality], method: str = "text") -> ExtractionQuality:
    report = ExtractionQuality(sections=list(qualities), method=method)
    if qualities:
        # the weakest report determines how far we can trust the extraction
        report.score = round(min(q.score for q in qualities), 2)
        report.issues = list(dict.fromkeys(f"{q.title}: {i}" for q in qualities for i in q.issues))
    report.low_confidence = report.score < CONFIDENCE_THRESHOLD
    report.level = "LOW" if report.low_confidence else ("MEDIUM" if report.score < 0.8 else "HIGH")
    return report


# ---------------------------------------------------------------------- uploads
_FREE_LAB = re.compile(
    r"^\s*(?P<param>[A-Za-z][A-Za-z0-9 ()/.,%+-]{1,40}?)\s*[:=-]?\s+(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-zµμ/%^][A-Za-z0-9µμ/%^.]*)?\s*[\[(]?\s*(?:ref(?:erence)?(?: range)?|normal(?: range)?|range)?[:\s]*"
    r"(?P<range>\d[\d,]*(?:\.\d+)?\s*(?:-|–|to)\s*\d[\d,]*(?:\.\d+)?|[<>≤≥]\s*\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit2>[A-Za-zµμ/%^][A-Za-z0-9µμ/%^.]*)?\s*[\])]?\s*(?P<flag>high|low|normal|h|l)?\s*$", re.I)


def free_text_labs(text: str) -> List[LabResult]:
    """Lab values typed as plain lines, e.g. 'Hemoglobin 9.2 g/dL 13.0-17.0' or 'WBC: 14500 /uL (ref 4000-11000)'."""
    out: List[LabResult] = []
    for line in text.splitlines():
        m = _FREE_LAB.match(line)
        if not m or re.search(r"\b(age|date|id|phone|mrn)\b", m.group("param"), re.I):
            continue
        unit2, flag = m.group("unit2") or "", (m.group("flag") or "").upper()
        if unit2.lower() in ("h", "l", "high", "low", "normal"):
            unit2, flag = "", unit2.upper()
        flag = {"H": "HIGH", "L": "LOW"}.get(flag, flag)
        r = make_lab_result(m.group("param"), m.group("value"), m.group("unit") or unit2, m.group("range"), flag,
                            origin="free-text")
        if r:
            out.append(r)
    return out


def parse_upload(path: Path, *, patient_key: str, patient_name: str, report_type: str = "auto", uploaded_by: str = "",
                 uploaded_at: str = "", display_name: Optional[str] = None, ocr_enabled: bool = True,
                 cache_dir: Optional[Path] = None) -> ClinicalDocument:
    """Turn an uploaded file into a ClinicalDocument whose sections all belong to `patient_key`."""
    if path.suffix.lower() not in SUPPORTED_UPLOADS:
        raise ValueError(f"Unsupported file type {path.suffix}. Supported: {', '.join(SUPPORTED_UPLOADS)}")
    blocks = load_document(path, ocr_enabled=ocr_enabled, cache_dir=cache_dir)
    if not blocks or not any(b.text.strip() for b in blocks):
        raise ValueError("No readable text could be extracted from this file")
    filename = display_name or path.name
    title = " ".join(Path(filename).stem.replace("_", " ").split()) or "Uploaded report"
    drafts = split_sections(blocks, title)
    for d in drafts:
        if not d.lab_results:
            d.lab_results = free_text_labs(d.text)
    text = "\n".join(d.text for d in drafts)
    if report_type in REPORT_TYPES:
        category, domain = REPORT_TYPES[report_type]
    else:
        labs = [r for d in drafts for r in d.lab_results]
        category, domain = classify(text, len(labs), bool(labs) and all("ige" in r.parameter.lower() for r in labs))
    digest = hashlib.sha1(path.read_bytes() + patient_key.encode()).hexdigest()[:10]
    doc_id = f"UPLOAD-{digest}"
    doc = ClinicalDocument(doc_id=doc_id, filename=filename, relative_path=f"uploads/{patient_key}/{filename}",
                           category=category, domain=domain, title=title, origin="upload",
                           extraction_method="ocr" if any(b.source == "ocr" for b in blocks) else "text",
                           uploaded_by=uploaded_by, uploaded_at=uploaded_at)
    for i, d in enumerate(drafts):
        if not d.text.strip():
            continue
        doc.sections.append(Section(
            section_id=f"{doc_id}-S{i:02d}", doc_id=doc_id,
            title=d.title if d.title != title or len(drafts) == 1 else f"{title} ({i + 1})",
            text=d.text, patient_key=patient_key, patient_name=patient_name, fields=d.fields,
            lab_results=d.lab_results, report_date=date_from_fields(d.fields) or date_from_text(d.text),
        ))
    if not doc.sections:
        raise ValueError("No readable text could be extracted from this file")
    return doc


def assess_document(doc: ClinicalDocument, expected_name: str) -> ExtractionQuality:
    return summarise_quality([assess_section(s, doc, expected_name) for s in doc.sections], method=doc.extraction_method)


# ---------------------------------------------------------------------- question-time extraction
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

    def by_domain(self, engine: "ClinicalEngine", domain: Domain) -> List[Section]:
        return [s for s in self.sections if engine.store.domain_of(s) == domain]

    def lab_count(self) -> int:
        return sum(len(s.lab_results) for s in self.sections)


class DataExtractionAgent:
    def __init__(self, engine: "ClinicalEngine"):
        self.engine = engine

    def run(self, question: str, role: Role, patient_key: Optional[str], focus: Sequence[Domain],
            new_symptoms: Sequence[str] = (), upload_doc_ids: Sequence[str] = ()) -> ExtractionResult:
        e, pol = self.engine, policy(role)
        res = ExtractionResult(patient_key=patient_key, new_symptoms=list(new_symptoms))
        res.denied = [d for d in focus if d not in pol.domains]
        res.focus = [d for d in focus if d in pol.domains]
        visible = [d for d in Domain if d in pol.domains and d != Domain.REFERENCE]
        wanted = res.focus or visible
        patient = e.patient(patient_key)

        if patient_key:
            sections = [s for s in e.store.patient_sections(patient_key, wanted)]
            if upload_doc_ids:
                sections = [s for s in sections if s.doc_id in set(upload_doc_ids)] or sections
            else:
                sections = self._focus(question, sections)
            res.sections = sections
            res.symptoms = list(dict.fromkeys([*e.kg.patient_symptoms(patient_key), *new_symptoms]))
            on_file = set(e.patient_domains(patient_key))
            for d in res.focus:
                if d != Domain.REFERENCE and d not in on_file:
                    res.missing.append(f"No {d.value.replace('_', ' ')} reports on file")
            if not sections and not res.missing:
                res.missing.append("No diagnostic reports on file — upload any reports you have (📎)")
            if patient and patient.age is None:
                res.missing.append("Age not recorded (age-specific thresholds cannot be applied)")
        else:
            hits = e.store.search(question, k=6, domains=[*wanted, Domain.REFERENCE] if res.focus else None)
            res.sections = [s for s in e.store.sections_for(hits, max_sections=3)
                            if e.store.domain_of(s) in pol.domains]
            res.reference_only = True
            res.symptoms = list(new_symptoms)

        qualities = [assess_section(s, e.store.document_of(s), patient.name if patient else None)
                     for s in res.sections if e.store.domain_of(s) != Domain.REFERENCE]
        res.quality = summarise_quality(qualities)
        res.sources = list(dict.fromkeys(e.store.source_of(s) for s in res.sections))
        res.sufficient = bool(res.sections) or bool(res.symptoms)
        return res

    def _focus(self, question: str, sections: List[Section]) -> List[Section]:
        q = question.lower()
        panels = {w for rx, words in LAB_PANELS.items() if re.search(rx, q) for w in words}
        modal = [t for t, rx in IMAGING_TERMS.items() if re.search(rx, q)]
        if not panels and not modal:
            return sections
        keep = []
        for s in sections:
            head = (s.title + " " + self.engine.store.source_of(s)).lower()
            if panels and s.lab_results and any(w in head for w in panels):
                keep.append(s)
            elif modal and not s.lab_results and sum(bool(re.search(IMAGING_TERMS[t], (s.title + " " + s.text[:300]).lower())) for t in modal):
                keep.append(s)
        return keep or sections
