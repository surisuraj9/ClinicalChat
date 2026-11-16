"""Split clinical documents into per-patient sections and extract structured fields."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from clinicalchat.domain.models import LabResult
from clinicalchat.ingestion.lab_tables import parse_lab_table, parse_pipe_lines
from clinicalchat.ingestion.loaders import Block

_NAME_RE = re.compile(
    r"^(?:Patient\s*Name|Name|Patient)\s*[:\-]\s*"
    r"(?!Dr\.?\s)([A-Z][a-zA-Z.'’]+(?:\s+[A-Z][a-zA-Z.'’]+){0,3})"
    r"(?:\s*,\s*(\d{1,3})\s*([MF])\b)?"
)
_HEADER_RE = re.compile(
    r"^(?:"
    r"Patient\s*\d+\s*[:–\-]"                       # Patient 1: CBC Report / Patient 2 – Non-Asthma
    r"|CASE\s*\d+\b"
    r"|SAMPLE\s+REPORT\s*\d*"
    r"|Discharge\s+Summary\s*\d+"
    r"|(?:\d+[A-Z]?[.)]\s*|[A-Z]\s*[–\-]\s*)?(?:AB)?NORMAL\s+(?:PATIENT\s+)?REPORT"
    r"|(?:[A-Z]\.\s*|VERY\s+)?DETAILED\s+ANALYSIS"
    r"|General\s+Analysis\b"
    r"|How\s+\w+\s+Works\b"
    r")",
    re.IGNORECASE,
)
# Upper-case only: "ABNORMAL MRI SPINE" is a header, "Abnormal ultrasound" inside a report is not.
_IMAGING_HEADER_RE = re.compile(r"^(?:AB)?NORMAL\s+(?:MRI|CT|X-?RAY|ULTRASOUND)\b")
_FIELD_RE = re.compile(r"^([A-Z][A-Za-z /()&\-]{1,40}?)\s*:\s*(.+)$")
_AGE_RE = re.compile(r"^(?:Age(?:/Sex)?)\s*:\s*(\d{1,3})(?:\s*(?:years?|yrs?|Y))?(?:\s*[/,]\s*(Male|Female|M|F))?", re.IGNORECASE)
_GENDER_RE = re.compile(r"^(?:Gender|Sex)\s*:\s*(Male|Female|M|F)\b", re.IGNORECASE)
_ID_RE = re.compile(r"^(?:Patient\s*ID|Hospital\s*No\.?)\s*:\s*(\S+)", re.IGNORECASE)


@dataclass
class _Draft:
    title: str = ""
    lines: List[str] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)

    def has_content(self) -> bool:
        return bool(self.lines or self.tables)


@dataclass
class SectionDraft:
    title: str
    text: str
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    identifiers: List[str]
    fields: Dict[str, str]
    lab_results: List[LabResult]


def _is_heading_like(line: str) -> bool:
    return len(line) <= 60 and ":" not in line and not line.startswith(("•", "-", "*")) and not line[0].isdigit()


def _flatten(blocks: List[Block]) -> List[Tuple[str, object]]:
    items: List[Tuple[str, object]] = []
    for block in blocks:
        if block.kind == "table":
            items.append(("table", block.rows))
        else:
            for line in block.text.splitlines():
                line = line.strip()
                if line:
                    items.append(("line", line))
    return items


def _gender(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return "Male" if value.upper().startswith("M") else "Female"


def _new_patient_draft(drafts: List[_Draft], current: _Draft, name: str, doc_title: str) -> _Draft:
    """A new patient without an explicit header: pull the trailing heading lines along into a new section."""
    carry: List[str] = []
    while current.lines and _is_heading_like(current.lines[-1]) and len(carry) < 3:
        carry.insert(0, current.lines.pop())
    draft = _Draft(title=carry[0] if carry else f"{doc_title} – {name}", lines=carry)
    drafts.append(draft)
    return draft


def _on_header(drafts: List[_Draft], current: _Draft, line: str) -> None:
    if current.has_content() or current.name:
        current = _Draft(title=line)
        drafts.append(current)
    current.title = line
    current.lines.append(line)


def _on_name(drafts: List[_Draft], current: _Draft, m: re.Match, line: str, doc_title: str) -> None:
    if current.name and current.name != m.group(1):
        current = _new_patient_draft(drafts, current, m.group(1), doc_title)
    if current.title == doc_title and current.lines and _is_heading_like(current.lines[-1]):
        current.title = current.lines[-1]
    current.name = m.group(1).strip()
    if m.group(2):
        current.age = int(m.group(2))
        current.gender = _gender(m.group(3))
    current.lines.append(line)


def _on_demographics(current: _Draft, line: str) -> None:
    age = _AGE_RE.match(line)
    if age and current.age is None:
        current.age = int(age.group(1))
        current.gender = current.gender or _gender(age.group(2))
    gender = _GENDER_RE.match(line)
    if gender and current.gender is None:
        current.gender = _gender(gender.group(1))
    ident = _ID_RE.match(line)
    if ident:
        current.identifiers.append(ident.group(1))


def _to_section(d: _Draft, doc_title: str) -> SectionDraft:
    fields: Dict[str, str] = {}
    for line in d.lines:
        m = _FIELD_RE.match(line)
        if m and m.group(1) not in fields:
            fields[m.group(1).strip()] = m.group(2).strip()
    labs: List[LabResult] = [r for rows in d.tables for r in parse_lab_table(rows)] + parse_pipe_lines(d.lines)
    text = "\n".join([*d.lines, *("\n".join(" | ".join(c for c in row if c) for row in rows) for rows in d.tables)])
    return SectionDraft(title=d.title or doc_title, text=text, name=d.name, age=d.age, gender=d.gender,
                        identifiers=d.identifiers, fields=fields, lab_results=labs)


def split_sections(blocks: List[Block], doc_title: str) -> List[SectionDraft]:
    """Cut a document into sections: a new section at every report header or when a new patient name appears."""
    drafts: List[_Draft] = [_Draft(title=doc_title)]
    for kind, payload in _flatten(blocks):
        current = drafts[-1]
        if kind == "table":
            current.tables.append(payload)  # type: ignore[arg-type]
            continue
        line: str = payload  # type: ignore[assignment]
        name_match = _NAME_RE.match(line)
        if _HEADER_RE.match(line) or _IMAGING_HEADER_RE.match(line):
            _on_header(drafts, current, line)
        elif name_match:
            _on_name(drafts, current, name_match, line, doc_title)
        else:
            if current.name:
                _on_demographics(current, line)
            current.lines.append(line)
    return [_to_section(d, doc_title) for d in drafts if d.has_content()]
