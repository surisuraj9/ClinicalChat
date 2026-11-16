"""Uploaded reports: read the file (DOCX/PDF/TXT/XLSX/image with OCR), find lab values, decide what kind of report it
is, and turn it into a ClinicalDocument that belongs to the uploading patient."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

from clinicalchat.domain.models import ClinicalDocument, Domain, LabResult, Section
from clinicalchat.ingestion.dates import date_from_fields, date_from_text
from clinicalchat.ingestion.lab_tables import make_lab_result, parse_pipe_lines
from clinicalchat.ingestion.loaders import load_document
from clinicalchat.ingestion.sectioning import split_sections

SUPPORTED_UPLOADS = [".pdf", ".docx", ".txt", ".md", ".xlsx", ".png", ".jpg", ".jpeg"]
NO_TEXT = "No readable text could be extracted from this file"
MAX_UPLOAD_FILES = 10  # per message, so each report can be checked properly

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


def classify(text: str, lab_count: int, allergy_panel: bool = False) -> Tuple[str, Domain]:
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


def _read_blocks(path: Path, ocr_enabled: bool, cache_dir: Optional[Path]) -> list:
    if path.suffix.lower() not in SUPPORTED_UPLOADS:
        raise ValueError(f"Unsupported file type {path.suffix}. Supported: {', '.join(SUPPORTED_UPLOADS)}")
    blocks = load_document(path, ocr_enabled=ocr_enabled, cache_dir=cache_dir)
    if not blocks or not any(b.text.strip() for b in blocks):
        raise ValueError(NO_TEXT)
    return blocks


def _report_kind(report_type: str, drafts: list) -> Tuple[str, Domain]:
    """The type the patient chose, else detected from the text (an all-IgE panel is an allergy test)."""
    if report_type in REPORT_TYPES:
        return REPORT_TYPES[report_type]
    labs = [r for d in drafts for r in d.lab_results]
    text = "\n".join(d.text for d in drafts)
    return classify(text, len(labs), bool(labs) and all("ige" in r.parameter.lower() for r in labs))


def reparse_generic_upload(doc: ClinicalDocument) -> bool:
    """Re-read lab values from an upload that was filed as a generic record with none (e.g. stored by an older
    parser), and re-classify it. Returns True when the document changed."""
    if doc.origin != "upload" or doc.domain != Domain.PATIENT_RECORD or any(s.lab_results for s in doc.sections):
        return False
    for s in doc.sections:
        s.lab_results = parse_pipe_lines(s.text.splitlines()) or free_text_labs(s.text)
    labs = [r for s in doc.sections for r in s.lab_results]
    if not labs:
        return False
    text = "\n".join(s.text for s in doc.sections)
    doc.category, doc.domain = classify(text, len(labs), all("ige" in r.parameter.lower() for r in labs))
    return True


def _sections(drafts: list, doc_id: str, title: str, patient_key: str, patient_name: str) -> List[Section]:
    return [Section(section_id=f"{doc_id}-S{i:02d}", doc_id=doc_id,
                    title=d.title if d.title != title or len(drafts) == 1 else f"{title} ({i + 1})",
                    text=d.text, patient_key=patient_key, patient_name=patient_name, fields=d.fields,
                    lab_results=d.lab_results, report_date=date_from_fields(d.fields) or date_from_text(d.text))
            for i, d in enumerate(drafts) if d.text.strip()]


def parse_upload(path: Path, *, patient_key: str, patient_name: str, report_type: str = "auto", uploaded_by: str = "",
                 uploaded_at: str = "", display_name: Optional[str] = None, ocr_enabled: bool = True,
                 cache_dir: Optional[Path] = None) -> ClinicalDocument:
    """Turn an uploaded file into a ClinicalDocument whose sections all belong to `patient_key`."""
    blocks = _read_blocks(path, ocr_enabled, cache_dir)
    filename = display_name or path.name
    title = " ".join(Path(filename).stem.replace("_", " ").split()) or "Uploaded report"
    drafts = split_sections(blocks, title)
    for d in drafts:
        d.lab_results = d.lab_results or free_text_labs(d.text)
    category, domain = _report_kind(report_type, drafts)
    doc_id = "UPLOAD-" + hashlib.sha1(path.read_bytes() + patient_key.encode()).hexdigest()[:10]
    doc = ClinicalDocument(doc_id=doc_id, filename=filename, relative_path=f"uploads/{patient_key}/{filename}",
                           category=category, domain=domain, title=title, origin="upload",
                           extraction_method="ocr" if any(b.source == "ocr" for b in blocks) else "text",
                           uploaded_by=uploaded_by, uploaded_at=uploaded_at,
                           sections=_sections(drafts, doc_id, title, patient_key, patient_name))
    if not doc.sections:
        raise ValueError(NO_TEXT)
    return doc
