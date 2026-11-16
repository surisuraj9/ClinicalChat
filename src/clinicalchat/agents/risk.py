"""🚨 Risk Detection Agent (deterministic, auditable rules — no LLM).

Rules: critical/panic lab values, test-specific severity thresholds (stricter for age >= 65), red-flag symptoms,
imaging red flags, contraindicated drugs, severe allergy history, multiple concurrent conditions, clinical
inconsistencies and low-confidence extraction. HIGH/CRITICAL risk or an explicit rule escalates the case for
clinician validation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from clinicalchat.agents.analysis import AnalysisResult
from clinicalchat.agents.extraction import ExtractionResult
from clinicalchat.agents.specialists.radiology import ImagingFinding
from clinicalchat.domain.models import LabResult, LabStatus, PatientProfile, SafetyAlert
from clinicalchat.domain.risk import ESCALATING_LEVELS, LEVEL_TO_URGENCY, LEVELS, RiskAssessment, RiskIndicator, bump
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph

_WEIGHT = {"LOW": 3, "MODERATE": 10, "HIGH": 25, "CRITICAL": 45}

# canonical test -> unit guard (regex on unit, "" = any) and ordered (op, threshold, level, reason)
THRESHOLDS: Dict[str, Tuple[str, List[Tuple[str, float, str, str]]]] = {
    "Hemoglobin": (r"g/?dl|^$", [("<", 7.0, "CRITICAL", "severe anaemia"), ("<", 8.0, "HIGH", "marked anaemia"),
                                 ("<", 10.0, "MODERATE", "moderate anaemia")]),
    "eGFR": (r"ml|^$", [("<", 15.0, "CRITICAL", "kidney failure range (G5)"), ("<", 30.0, "HIGH", "severely reduced kidney function (G4)"),
                        ("<", 60.0, "MODERATE", "reduced kidney function (G3)")]),
    "Potassium": (r"mmol|meq|^$", [(">=", 6.0, "CRITICAL", "severe hyperkalaemia"), (">=", 5.5, "HIGH", "hyperkalaemia"),
                                   ("<", 3.0, "HIGH", "significant hypokalaemia")]),
    "Sodium": (r"mmol|meq|^$", [("<", 125.0, "CRITICAL", "severe hyponatraemia"), ("<", 130.0, "HIGH", "hyponatraemia"),
                                (">", 155.0, "CRITICAL", "severe hypernatraemia")]),
    "HbA1c": (r"%|^$", [(">=", 10.0, "HIGH", "poorly controlled diabetes"), (">=", 6.5, "MODERATE", "diabetes-range HbA1c")]),
    "Fasting glucose": (r"mg|^$", [(">=", 300.0, "HIGH", "marked hyperglycaemia"), ("<", 54.0, "CRITICAL", "clinically significant hypoglycaemia"),
                                   (">=", 126.0, "MODERATE", "diabetes-range fasting glucose")]),
    "TSH": (r"", [(">", 20.0, "HIGH", "overt hypothyroidism"), (">", 10.0, "MODERATE", "hypothyroidism needing treatment")]),
    "Troponin": (r"", [(">", -1.0, "HIGH", "elevated troponin — myocardial injury")]),
    "CRP": (r"mg/?l$|^$", [(">", 100.0, "HIGH", "very high CRP — possible serious bacterial infection"),
                           (">", 10.0, "MODERATE", "raised CRP — active infection or inflammation")]),
    "WBC count": (r"", [(">=", 25000.0, "HIGH", "marked leucocytosis"),
                        ("<", 2000.0, "HIGH", "marked leucopenia — infection risk")]),
}
ULN_RULES = {"ALT": (3.0, "HIGH", "ALT >3x upper limit"), "AST": (3.0, "HIGH", "AST >3x upper limit"),
             "Bilirubin": (3.0, "HIGH", "bilirubin >3x upper limit")}
AGE_SENSITIVE = {"eGFR", "Hemoglobin", "Potassium", "Sodium"}

RED_FLAG_SYMPTOMS = {
    "chest pain": ("CRITICAL", "Chest pain — possible heart attack; needs emergency evaluation"),
    "weakness on one side": ("CRITICAL", "One-sided weakness / slurred speech — possible stroke; emergency"),
    "throat tightness": ("CRITICAL", "Throat tightness — possible anaphylaxis / airway compromise; emergency"),
    "confusion": ("CRITICAL", "New confusion — needs emergency evaluation"),
    "coughing blood": ("HIGH", "Coughing up blood — needs same-day assessment"),
    "fainting": ("HIGH", "Fainting / loss of consciousness — needs same-day assessment"),
    "shortness of breath": ("HIGH", "Shortness of breath — needs prompt assessment"),
    "breathlessness": ("HIGH", "Breathlessness — needs prompt assessment"),
    "jaundice": ("HIGH", "Jaundice — needs prompt assessment of liver / biliary disease"),
}

def _cmp(op: str, a: float, b: float) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _threshold_rule(test: str, r: LabResult) -> Optional[Tuple[str, str]]:
    """(level, reason) from the test-specific thresholds or the "x times the upper limit" rules."""
    rule = THRESHOLDS.get(test)
    if test == "WBC count" and r.value < 200:  # reported in thousands (e.g. 14.8 x10^3/uL); thresholds are per µL
        rule = None
    troponin_not_high = test == "Troponin" and r.status not in (LabStatus.HIGH, LabStatus.CRITICAL_HIGH)
    if rule and not troponin_not_high and re.search(rule[0], (r.unit or "").strip().lower()):
        for op, thr, level, reason in rule[1]:
            if _cmp(op, r.value, thr):
                return level, reason
    if test in ULN_RULES and r.ref_high and r.value > ULN_RULES[test][0] * r.ref_high:
        return ULN_RULES[test][1], ULN_RULES[test][2]
    return None


def lab_indicators(results: Sequence[Tuple[LabResult, str]], age: Optional[int]) -> List[RiskIndicator]:
    out: List[RiskIndicator] = []
    other_abnormal = 0
    for r, source in results:
        if r.value is None:
            continue
        if r.status in (LabStatus.CRITICAL_LOW, LabStatus.CRITICAL_HIGH):
            out.append(RiskIndicator(indicator=f"Critical value: {r.line()}", level="CRITICAL", source=source, escalate=True))
            continue
        test = MedicalKnowledgeGraph.canonical_test(r.parameter) or r.parameter
        hit = _threshold_rule(test, r)
        if hit is None:
            other_abnormal += r.abnormal
            continue
        level, reason = hit
        if age is not None and age >= 65 and test in AGE_SENSITIVE and level != "CRITICAL":
            level, reason = bump(level), reason + " (age >= 65: stricter threshold)"
        out.append(RiskIndicator(indicator=f"{reason}: {r.line()}", level=level, source=source,
                                 escalate=level in ESCALATING_LEVELS))
    if other_abnormal:
        out.append(RiskIndicator(indicator=f"{other_abnormal} other value(s) outside the reference range", level="LOW"))
    return out


def symptom_indicators(symptoms: Sequence[str], current: Optional[Sequence[str]] = None) -> List[RiskIndicator]:
    """Red flags reported now keep their full level; earlier ones (already acted on) are kept as a MODERATE reminder."""
    out = []
    for s in dict.fromkeys(symptoms):
        if s in RED_FLAG_SYMPTOMS:
            level, why = RED_FLAG_SYMPTOMS[s]
            if current is not None and s not in current:
                out.append(RiskIndicator(indicator=f"Previously reported red-flag symptom: {s} — confirm it was assessed",
                                         level="MODERATE", category="symptom", source="Symptom history", escalate=True))
                continue
            out.append(RiskIndicator(indicator=f"Red-flag symptom: {why}", level=level, category="symptom",
                                     source="Reported symptoms", escalate=True))
    return out


def imaging_indicators(imaging: Sequence[ImagingFinding]) -> List[RiskIndicator]:
    out = []
    for f in imaging:
        title = f.section.title
        if f.red_flags:
            out.append(RiskIndicator(indicator=f"Imaging red flag: {'; '.join(h[:100] for h in f.red_flags[:2])} ({title})",
                                     level="HIGH", category="imaging", source=title, escalate=True))
        elif f.abnormal:
            out.append(RiskIndicator(indicator=f"Abnormal imaging: {'; '.join(h[:100] for h in f.abnormal[:2])} ({title})",
                                     level="MODERATE", category="imaging", source=title))
    return out


def medication_indicators(proposed: Sequence[SafetyAlert], current: Sequence[SafetyAlert]) -> List[RiskIndicator]:
    out = []
    for a in proposed:
        if a.level == "CONTRAINDICATED":
            out.append(RiskIndicator(indicator=f"{a.drug} is contraindicated ({a.against}) — do not give", level="HIGH",
                                     category="medication", escalate=True))
        elif a.level == "MAJOR":
            out.append(RiskIndicator(indicator=f"Major risk: {a.drug} ↔ {a.against}", level="MODERATE", category="medication"))
    for a in current:
        if a.level in ("CONTRAINDICATED", "MAJOR"):
            out.append(RiskIndicator(indicator=f"Current medicines: {a.drug} ↔ {a.against} ({a.level.lower()}): {a.message}",
                                     level="HIGH" if a.level == "CONTRAINDICATED" else "MODERATE", category="medication",
                                     escalate=a.level == "CONTRAINDICATED"))
    return out


def allergy_history_indicators(patient: Optional[PatientProfile]) -> List[RiskIndicator]:
    return [RiskIndicator(indicator=f"History of {a.severity.lower()} reaction to {a.allergen} — keep emergency plan ready",
                          level="MODERATE", category="allergy", source=a.source)
            for a in (patient.allergies if patient else []) if a.severity in ("CRITICAL", "SEVERE")]


def pattern_indicators(conditions: Sequence[str], inconsistencies: Sequence[str],
                       low_confidence: bool) -> List[RiskIndicator]:
    out = []
    if len(conditions) >= 3:
        out.append(RiskIndicator(indicator=f"Multiple potential conditions ({', '.join(conditions[:6])}) — clinician validation",
                                 level="MODERATE", category="pattern", escalate=True))
    out += [RiskIndicator(indicator=f"Clinical inconsistency: {issue}", level="MODERATE", category="consistency")
            for issue in inconsistencies]
    if low_confidence:
        out.append(RiskIndicator(indicator="Low-confidence data extraction — verify against the source report",
                                 level="MODERATE", category="extraction", escalate=True))
    return out


def assessment(indicators: List[RiskIndicator]) -> RiskAssessment:
    """The overall level is the most severe indicator; any escalating indicator (or HIGH/CRITICAL) escalates."""
    ind = sorted(indicators, key=lambda i: -LEVELS.index(i.level))
    level = ind[0].level if ind else "LOW"
    reasons = list(dict.fromkeys(i.indicator for i in ind if i.escalate or i.level in ESCALATING_LEVELS))
    return RiskAssessment(risk_level=level, urgency=LEVEL_TO_URGENCY[level],
                          score=min(100, sum(_WEIGHT[i.level] for i in ind)),
                          escalate=any(i.escalate for i in ind) or level in ESCALATING_LEVELS,
                          escalation_reasons=reasons[:8], indicators=ind)


class RiskDetectionAgent:
    def run(self, patient: Optional[PatientProfile], ext: ExtractionResult, an: AnalysisResult) -> RiskAssessment:
        return assessment([
            *lab_indicators(an.lab_results, patient.age if patient else None),
            *symptom_indicators(ext.symptoms, ext.new_symptoms),
            *imaging_indicators(an.imaging),
            *medication_indicators(an.drug_alerts, an.medication_alerts),
            *allergy_history_indicators(patient),
            *pattern_indicators(an.possible_conditions, an.inconsistencies, ext.quality.low_confidence),
        ])
