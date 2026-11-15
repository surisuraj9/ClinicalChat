"""Split clinical documents into per-patient sections and extract structured fields."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from clinicalchat.ingestion.lab_tables import parse_lab_table, parse_pipe_lines
from clinicalchat.ingestion.loaders import Block
from clinicalchat.models import LabResult

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


def split_sections(blocks: List[Block], doc_title: str) -> List[SectionDraft]:
    drafts: List[_Draft] = [_Draft(title=doc_title)]

    def start_new(title: str, carry: Optional[List[str]] = None) -> _Draft:
        draft = _Draft(title=title, lines=list(carry or []))
        drafts.append(draft)
        return draft

    for kind, payload in _flatten(blocks):
        current = drafts[-1]
        if kind == "table":
            current.tables.append(payload)  # type: ignore[arg-type]
            continue
        line: str = payload  # type: ignore[assignment]

        if _HEADER_RE.match(line) or _IMAGING_HEADER_RE.match(line):
            current = start_new(line) if current.has_content() or current.name else current
            current.title = line
            current.lines.append(line)
            continue

        name_match = _NAME_RE.match(line)
        if name_match:
            if current.name and current.name != name_match.group(1):
                # a new patient without an explicit header: pull trailing heading lines along
                carry: List[str] = []
                while current.lines and _is_heading_like(current.lines[-1]) and len(carry) < 3:
                    carry.insert(0, current.lines.pop())
                current = start_new(carry[0] if carry else f"{doc_title} – {name_match.group(1)}", carry)
            if current.title == doc_title and current.lines and _is_heading_like(current.lines[-1]):
                current.title = current.lines[-1]
            current.name = name_match.group(1).strip()
            if name_match.group(2):
                current.age = int(name_match.group(2))
                current.gender = _gender(name_match.group(3))
            current.lines.append(line)
            continue

        if current.name:
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
        current.lines.append(line)

    sections: List[SectionDraft] = []
    for d in drafts:
        if not d.has_content():
            continue
        fields: Dict[str, str] = {}
        for line in d.lines:
            m = _FIELD_RE.match(line)
            if m and m.group(1) not in fields:
                fields[m.group(1).strip()] = m.group(2).strip()
        labs: List[LabResult] = []
        for rows in d.tables:
            labs.extend(parse_lab_table(rows))
        labs.extend(parse_pipe_lines(d.lines))
        text_parts = list(d.lines)
        for rows in d.tables:
            text_parts.append("\n".join(" | ".join(c for c in row if c) for row in rows))
        sections.append(
            SectionDraft(
                title=d.title or doc_title, text="\n".join(text_parts), name=d.name, age=d.age,
                gender=d.gender, identifiers=d.identifiers, fields=fields, lab_results=labs,
            )
        )
    return sections
