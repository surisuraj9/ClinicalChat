"""Deterministic guard on every patient-facing AI text: no doses, no definitive diagnosis, no "stop your medicine",
no over-reassurance when results are abnormal, and always a pointer to the doctor."""

from __future__ import annotations

import re
from typing import Iterable, List

# a lab concentration ("15 g/dL", "126 mg per decilitre") is a result, not a dose
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|units?|iu)\b"
                   r"(?!\s*(?:/|per\s+)(?:dl|l|ml|deciliter|decilitre|liter|litre|cu\s?mm|mm3|ul|µl)\b)"
                   r"(?:\s*(?:once|twice|daily|a day|per day|bd|od|tds))?", re.I)
_DEFINITIVE = re.compile(r"\byou (?:definitely |certainly )?(?:have|are suffering from|are diagnosed with) "
                         r"(?:cancer|a tumou?r|leukaemia|leukemia|kidney failure|heart failure|a stroke|diabetes|hiv|pneumonia)\b", re.I)
_SELF_MED = re.compile(r"\b(stop|start|double|increase|decrease) (?:taking )?(?:your |the )?(?:medicine|medication|tablets?|insulin|dose)\b", re.I)
_REASSURE = re.compile(r"\b(nothing to worry about|completely normal|no need to see (?:a|your) doctor)\b", re.I)

# issues that stop the text from being shown before a clinician checks it (a missing "see your doctor" does not)
BLOCKING_GUARD = ("doses", "definitive diagnosis", "changing medication", "Over-reassuring")


def check_patient_text(parts: Iterable[str], has_abnormal: bool) -> List[str]:
    text = "\n".join(p for p in parts if p)
    issues: List[str] = []
    if _DOSE.search(text):
        issues.append("Contains medication doses — patient messages must not include dosing")
    if _DEFINITIVE.search(text):
        issues.append("States a definitive diagnosis — must stay informational until a clinician confirms")
    if _SELF_MED.search(text):
        issues.append("Suggests changing medication without a clinician")
    if has_abnormal and _REASSURE.search(text):
        issues.append("Over-reassuring wording although abnormal findings are present")
    if "doctor" not in text.lower() and "clinician" not in text.lower():
        issues.append("Does not direct the patient to discuss with their doctor")
    return issues


def blocking(issues: Iterable[str]) -> List[str]:
    return [i for i in issues if any(k in i for k in BLOCKING_GUARD)]
