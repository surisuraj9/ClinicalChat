"""Radiology Analyzer: red flags, abnormal findings and the impression in imaging reports (negation-aware:
"no fracture" is not a fracture)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from clinicalchat.domain.models import Section

_IMAGING_RED_FLAGS = ["haemorrhage", "hemorrhage", "infarct", "pneumothorax", r"(?<!body )\bmass\b", "malignan", "metasta",
                      "fracture", "dissection", "stroke", "midline shift", "cord compression", "free air", "tumou?r",
                      "aneurysm", "carcinoma", "appendicitis", r"\bsuspicious\b", r"pi.?rads [45]"]
_ABNORMAL_IMAGING = re.compile(r"consolidation|effusion|pneumonia|pneumonitis|ground.?glass|cardiomegaly|nodule|lesion|"
                               r"opacit|infiltrat|edema|oedema|stenosis|herniat|collapse|atelecta|thicken|calcul|stone|"
                               r"cyst|abnormal|enlarg|degenerat|ischemi|ischaemi|hypodensit", re.I)
_NEGATION = re.compile(r"\b(no|without|negative for|free of|absence of|not seen|ruled out|rule out)\b[^.\n]{0,40}$", re.I)
_SKIP_HEADER = re.compile(r"normal values|normal range|threshold|classification|reference|why |analysis|guide|quantitative", re.I)
_SKIP_LINE = re.compile(r"→|->|\(normal|normal:|expected|\b0\s?(?:mm|cm)\b", re.I)
_INLINE_FIELD = re.compile(r"^[A-Za-z /&()\-]{2,40}:\s*\S")
_IMPRESSION = re.compile(r"(?:impression|conclusion)\s*:?\s*(.+)", re.I)


def _report_lines(text: str, title: str = ""):
    header = ""
    for line in text.splitlines():
        st = line.strip(" \t•-*")
        if not st or (title and st == title.strip()):
            continue
        if st.endswith(":") and len(st) < 60:
            header = st
            continue
        if _INLINE_FIELD.match(st):
            header = ""
        if _SKIP_HEADER.search(header) or _SKIP_LINE.search(st):
            continue
        yield header, st


def imaging_red_flags(text: str, title: str = "") -> List[str]:
    hits = []
    for _, st in _report_lines(text, title):
        low = st.lower()
        for flag in _IMAGING_RED_FLAGS:
            m = re.search(flag, low)
            if m and not _NEGATION.search(low[:m.start()][-60:]):
                hits.append(st)
                break
    return list(dict.fromkeys(hits))


def imaging_abnormal(text: str, title: str = "") -> List[str]:
    hits = []
    for header, st in _report_lines(text, title):
        if not re.search(r"finding|impression|diagnos|observation|conclusion|comparison", header + " " + st.split(":")[0], re.I):
            continue
        m = _ABNORMAL_IMAGING.search(st)
        if m and not _NEGATION.search(st[:m.start()][-60:]) and not re.search(r"\bno\b|\bnormal\b(?! limits)", st[:m.start()], re.I):
            hits.append(st)
    return list(dict.fromkeys(hits))


def impression_of(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _IMPRESSION.match(line.strip(" \t•-*"))
        if m:
            rest = m.group(1).strip()
            if not rest and i + 1 < len(lines):
                rest = lines[i + 1].strip(" \t•-*")
            return rest[:300]
    return ""


@dataclass
class ImagingFinding:
    section: Section
    source: str
    impression: str
    red_flags: List[str]
    abnormal: List[str]


def analyse_imaging(section: Section, source: str) -> ImagingFinding:
    return ImagingFinding(section=section, source=source, impression=impression_of(section.text),
                          red_flags=imaging_red_flags(section.text, section.title),
                          abnormal=imaging_abnormal(section.text, section.title))
