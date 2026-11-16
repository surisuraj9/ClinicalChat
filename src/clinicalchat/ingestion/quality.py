"""Extraction quality: how far can we trust what was read from a report? (score 0–1, issues, low-confidence flag)"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from clinicalchat.domain.models import ClinicalDocument, Domain, LabStatus, Section
from clinicalchat.ingestion.uploads import names_match, printed_name

CONFIDENCE_THRESHOLD = 0.55


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


def _report_type(doc: Optional[ClinicalDocument]) -> str:
    if doc is None:
        return "clinical_note"
    return {Domain.LAB: "lab", Domain.RADIOLOGY: "radiology", Domain.ALLERGY: "allergy",
            Domain.PATIENT_RECORD: "clinical_note", Domain.REFERENCE: "reference"}[doc.domain]


Penalty = Tuple[float, str]  # (score deduction, issue shown to people)


def _general_penalties(section: Section, doc: Optional[ClinicalDocument]) -> List[Penalty]:
    out: List[Penalty] = []
    text_len = len(section.text.strip())
    if text_len < 150:
        out.append((0.35, f"very little text extracted ({text_len} characters)"))
    if doc is not None and doc.extraction_method == "ocr":
        out.append((0.15, "text read by OCR from a scanned image — verify values against the original"))
    if not section.report_date:
        out.append((0.05, "no report/test date found"))
    return out


def _lab_penalties(section: Section) -> List[Penalty]:
    labs = section.lab_results
    if not labs:
        return [(0.35, "lab report but no structured values could be parsed")]
    out: List[Penalty] = []
    with_range = sum(1 for r in labs if r.ref_low is not None or r.ref_high is not None)
    if with_range / len(labs) < 0.7:
        out.append((0.2, f"reference ranges missing for {len(labs) - with_range} of {len(labs)} values"))
    unknown = sum(1 for r in labs if r.status == LabStatus.UNKNOWN)
    if unknown / len(labs) > 0.3:
        out.append((0.1, f"{unknown} values could not be classified against a range"))
    disc = [r.parameter for r in labs if r.status_discrepancy]
    if disc:
        out.append((min(0.15, 0.05 * len(disc)), "report flag disagrees with printed range: " + ", ".join(disc[:5])))
    return out


def _radiology_penalties(section: Section) -> List[Penalty]:
    low = section.text.lower()
    out: List[Penalty] = []
    if "impression" not in low and "conclusion" not in low:
        out.append((0.15, "no impression/conclusion section found"))
    if "finding" not in low and "observation" not in low:
        out.append((0.1, "no findings section found"))
    return out


_TYPE_PENALTIES = {"lab": _lab_penalties, "radiology": _radiology_penalties}


def assess_section(section: Section, doc: Optional[ClinicalDocument], expected_name: Optional[str] = None) -> SectionQuality:
    rtype = _report_type(doc)
    penalties = _general_penalties(section, doc)
    if rtype in _TYPE_PENALTIES:
        penalties += _TYPE_PENALTIES[rtype](section)
    printed = printed_name(section.text)
    if expected_name and printed and not names_match(printed, expected_name):
        penalties.append((0.5, f"name on report ('{printed}') does not match the patient ('{expected_name}')"))
    return SectionQuality(section_id=section.section_id, title=section.title, source=doc.filename if doc else "",
                          report_type=rtype, report_date=section.report_date, lab_values=len(section.lab_results),
                          abnormal_values=sum(1 for r in section.lab_results if r.abnormal),
                          score=round(max(0.0, 1.0 - sum(p for p, _ in penalties)), 2),
                          issues=[msg for _, msg in penalties])


def summarise_quality(qualities: Sequence[SectionQuality], method: str = "text") -> ExtractionQuality:
    report = ExtractionQuality(sections=list(qualities), method=method)
    if qualities:
        # the weakest report determines how far we can trust the extraction
        report.score = round(min(q.score for q in qualities), 2)
        report.issues = list(dict.fromkeys(f"{q.title}: {i}" for q in qualities for i in q.issues))
    report.low_confidence = report.score < CONFIDENCE_THRESHOLD
    report.level = "LOW" if report.low_confidence else ("MEDIUM" if report.score < 0.8 else "HIGH")
    return report


def assess_document(doc: ClinicalDocument, expected_name: str) -> ExtractionQuality:
    return summarise_quality([assess_section(s, doc, expected_name) for s in doc.sections], method=doc.extraction_method)
