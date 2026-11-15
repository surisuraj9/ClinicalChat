"""🚨 Risk Detection Agent (deterministic, auditable rules — no LLM).

Rules: critical/panic lab values, test-specific severity thresholds (stricter for age >= 65), red-flag symptoms,
imaging red flags, contraindicated drugs, severe allergy history, multiple concurrent conditions, clinical
inconsistencies and low-confidence extraction. HIGH/CRITICAL risk or an explicit rule escalates the case for
clinician validation (slide negative scenario: "multiple potential conditions / false positive -> escalate").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from clinicalchat.knowledge_graph import MedicalKnowledgeGraph
from clinicalchat.models import LabResult, LabStatus, PatientProfile, SafetyAlert, Section

LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
CLINICAL_CATEGORIES = {"lab", "imaging", "symptom", "pattern"}
LEVEL_TO_URGENCY = {"LOW": "ROUTINE", "MODERATE": "SOON", "HIGH": "URGENT", "CRITICAL": "EMERGENCY"}
_WEIGHT = {"LOW": 3, "MODERATE": 10, "HIGH": 25, "CRITICAL": 45}

# canonical test -> unit guard (regex on unit, "" = any) and ordered (op, threshold, level, reason)
THRESHOLDS: Dict[str, tuple[str, List[tuple[str, float, str, str]]]] = {
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

PATIENT_GUIDANCE = {
    "CRITICAL": "🚑 **This could be an emergency. Call your local emergency number (e.g. 112 / 108 / 911) or go to the "
                "nearest emergency department now.** Do not wait for an online reply.",
    "HIGH": "🏥 **Please see a doctor today** (urgent care or your GP). If symptoms get worse, call emergency services.",
    "MODERATE": "📅 **Book an appointment with your doctor in the next few days.** Seek care sooner if you feel worse.",
    "LOW": "🗓️ **No urgent warning signs found.** Discuss these results at your next routine visit, or sooner if "
           "you feel unwell.",
}


class RiskIndicator(BaseModel):
    indicator: str
    level: str = "LOW"
    category: str = "lab"  # lab | medication | allergy | imaging | symptom | pattern | consistency | extraction
    source: str = ""
    escalate: bool = False


class RiskAssessment(BaseModel):
    risk_level: str = "LOW"
    urgency: str = "ROUTINE"
    score: int = 0
    escalate: bool = False
    escalation_reasons: List[str] = Field(default_factory=list)
    indicators: List[RiskIndicator] = Field(default_factory=list)

    @property
    def clinical_level(self) -> str:
        """Risk from the patient's condition (labs, imaging, symptoms) — drives patient guidance; a question about
        a contraindicated drug is a safety alert, not a sign that the patient is unwell."""
        levels = [i.level for i in self.indicators if i.category in CLINICAL_CATEGORIES]
        return max(levels, key=LEVELS.index) if levels else "LOW"

    @property
    def red_flag_symptoms(self) -> List[RiskIndicator]:
        return [i for i in self.indicators if i.category == "symptom"]


def _cmp(op: str, a: float, b: float) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _bump(level: str) -> str:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


def lab_indicators(results: Sequence[tuple[LabResult, str]], age: Optional[int]) -> List[RiskIndicator]:
    out: List[RiskIndicator] = []
    other_abnormal = 0
    for r, source in results:
        if r.value is None:
            continue
        test = MedicalKnowledgeGraph.canonical_test(r.parameter) or r.parameter
        if r.status in (LabStatus.CRITICAL_LOW, LabStatus.CRITICAL_HIGH):
            out.append(RiskIndicator(indicator=f"Critical value: {r.line()}", level="CRITICAL", source=source, escalate=True))
            continue
        hit = None
        rule = THRESHOLDS.get(test)
        if test == "WBC count" and r.value < 200:  # reported in thousands (e.g. 14.8 x10^3/uL); thresholds are per µL
            rule = None
        if rule and re.search(rule[0], (r.unit or "").strip().lower()):
            for op, thr, level, reason in rule[1]:
                if test == "Troponin" and r.status not in (LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
                    break
                if _cmp(op, r.value, thr):
                    hit = (level, reason)
                    break
        if hit is None and test in ULN_RULES and r.ref_high and r.value > ULN_RULES[test][0] * r.ref_high:
            hit = ULN_RULES[test][1:]
        if hit:
            level, reason = hit
            if age is not None and age >= 65 and test in AGE_SENSITIVE and level != "CRITICAL":
                level, reason = _bump(level), reason + " (age >= 65: stricter threshold)"
            out.append(RiskIndicator(indicator=f"{reason}: {r.line()}", level=level, source=source,
                                     escalate=level in ("HIGH", "CRITICAL")))
        elif r.abnormal:
            other_abnormal += 1
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


class RiskDetectionAgent:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg

    def run(self, *, patient: Optional[PatientProfile], lab_results: Sequence[tuple[LabResult, str]],
            imaging: Sequence[tuple[Section, List[str], List[str]]], symptoms: Sequence[str],
            new_symptoms: Optional[Sequence[str]] = None,
            proposed_drug_alerts: Sequence[SafetyAlert], medication_alerts: Sequence[SafetyAlert],
            conditions: Sequence[str], inconsistencies: Sequence[str], low_confidence: bool) -> RiskAssessment:
        ind: List[RiskIndicator] = []
        ind += lab_indicators(lab_results, patient.age if patient else None)
        ind += symptom_indicators(symptoms, new_symptoms)
        for section, red_flags, abnormal in imaging:
            if red_flags:
                ind.append(RiskIndicator(indicator=f"Imaging red flag: {'; '.join(h[:100] for h in red_flags[:2])} ({section.title})",
                                         level="HIGH", category="imaging", source=section.title, escalate=True))
            elif abnormal:
                ind.append(RiskIndicator(indicator=f"Abnormal imaging: {'; '.join(h[:100] for h in abnormal[:2])} ({section.title})",
                                         level="MODERATE", category="imaging", source=section.title))
        for a in proposed_drug_alerts:
            if a.level == "CONTRAINDICATED":
                ind.append(RiskIndicator(indicator=f"{a.drug} is contraindicated ({a.against}) — do not give", level="HIGH",
                                         category="medication", escalate=True))
            elif a.level == "MAJOR":
                ind.append(RiskIndicator(indicator=f"Major risk: {a.drug} ↔ {a.against}", level="MODERATE", category="medication"))
        for a in medication_alerts:
            if a.level in ("CONTRAINDICATED", "MAJOR"):
                ind.append(RiskIndicator(indicator=f"Current medicines: {a.drug} ↔ {a.against} ({a.level.lower()}): {a.message}",
                                         level="HIGH" if a.level == "CONTRAINDICATED" else "MODERATE", category="medication",
                                         escalate=a.level == "CONTRAINDICATED"))
        if patient:
            for a in patient.allergies:
                if a.severity in ("CRITICAL", "SEVERE"):
                    ind.append(RiskIndicator(indicator=f"History of {a.severity.lower()} reaction to {a.allergen} — keep emergency plan ready",
                                             level="MODERATE", category="allergy", source=a.source))
        if len(conditions) >= 3:
            ind.append(RiskIndicator(indicator=f"Multiple potential conditions ({', '.join(conditions[:6])}) — clinician validation",
                                     level="MODERATE", category="pattern", escalate=True))
        for issue in inconsistencies:
            ind.append(RiskIndicator(indicator=f"Clinical inconsistency: {issue}", level="MODERATE", category="consistency"))
        if low_confidence:
            ind.append(RiskIndicator(indicator="Low-confidence data extraction — verify against the source report",
                                     level="MODERATE", category="extraction", escalate=True))

        ind.sort(key=lambda i: -LEVELS.index(i.level))
        level = ind[0].level if ind else "LOW"
        escalate = any(i.escalate for i in ind) or level in ("HIGH", "CRITICAL")
        reasons = list(dict.fromkeys(i.indicator for i in ind if i.escalate or i.level in ("HIGH", "CRITICAL")))
        return RiskAssessment(risk_level=level, urgency=LEVEL_TO_URGENCY[level],
                              score=min(100, sum(_WEIGHT[i.level] for i in ind)), escalate=escalate,
                              escalation_reasons=reasons[:8], indicators=ind)
