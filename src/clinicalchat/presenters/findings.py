"""Short summaries of what the three rule-based agents found, for the live progress lines and the AI prompts."""

from __future__ import annotations

from typing import Optional, Sequence

from clinicalchat.agents.analysis import AnalysisResult
from clinicalchat.agents.extraction import ExtractionResult
from clinicalchat.domain.models import PatientProfile
from clinicalchat.domain.risk import LEVEL_BADGE, RiskAssessment
from clinicalchat.services.router import RoutePlan


def route_text(plan: RoutePlan, patient: Optional[PatientProfile]) -> str:
    who = patient.label if patient else "no specific patient"
    text = f"**Patient:** {who}" + (" (named in question)" if plan.patient_source == "question" else "")
    return text + (f"\n**Pipeline:** {plan.describe()}" if not plan.blocked else "")


def extraction_text(ext: ExtractionResult, is_patient: bool) -> str:
    if ext.sections:
        what = "your" if is_patient else "the"
        text = f"Selected {len(ext.sections)} section(s) from {what} records: {', '.join(ext.sources[:5])}"
        if not ext.reference_only:
            text += f" · extraction confidence **{ext.quality.level}**"
    else:
        text = "No matching reports on file"
    if ext.new_symptoms:
        text += "\n- 📝 Saved to " + ("your record" if is_patient else "the record") + ": " + ", ".join(ext.new_symptoms)
    text += "".join(f"\n- ℹ️ {m}" for m in ext.missing[:3])
    text += "".join(f"\n- ⚠️ {i}" for i in ext.quality.issues[:3])
    if ext.denied:
        text += "\n- 🔒 Not permitted for your role: " + ", ".join(d.value.replace("_", " ") for d in ext.denied)
    return text


def extraction_brief(ext: ExtractionResult) -> str:
    lines = [f"Records used: {', '.join(ext.sources) or 'none'}",
             f"Extraction quality: {ext.quality.level} ({ext.quality.score:.2f})"]
    if ext.symptoms:
        lines.append("Patient-reported symptoms: " + ", ".join(ext.symptoms))
    lines += [f"Missing: {m}" for m in ext.missing] + [f"Quality issue: {i}" for i in ext.quality.issues[:4]]
    return "\n".join(lines)


def _lab_line(an: AnalysisResult) -> list:
    if not an.lab_results:
        return []
    line = f"**Lab Interpreter:** {len(an.lab_results)} values checked, {len(an.abnormal)} outside the normal range"
    if an.abnormal:
        line += " — " + ", ".join(f"{r.parameter} {r.value_text} ({r.status.value.lower()})" for r, _ in an.abnormal[:6])
    return [line]


def _imaging_lines(an: AnalysisResult) -> list:
    out = []
    for f in an.imaging[:2]:
        notes = f.red_flags[:2] or f.abnormal[:2]
        tail = ("🚩 " if f.red_flags else "") + "; ".join(h[:90] for h in notes) if notes else "no abnormal finding flagged"
        out.append(f"**Radiology Analyzer:** {f.section.title} — {tail}")
    return out


def _allergy_line(an: AnalysisResult) -> list:
    if an.drug_alerts:
        return ["**Allergy Safety:** " + "; ".join(f"{a.level} — {a.drug} vs {a.against}" for a in an.drug_alerts)]
    if an.allergies:
        return ["**Allergy Safety:** documented allergies " + ", ".join(an.allergies)]
    return []


def analysis_text(an: AnalysisResult, is_patient: bool, medications: Sequence[str] = ()) -> str:
    out = _lab_line(an) + _imaging_lines(an) + _allergy_line(an)
    if an.differential:
        prefix = "**Symptoms may fit (not a diagnosis):** " if is_patient else "**Symptom differential (KG):** "
        out.append(prefix + ", ".join(str(d["condition"]) for d in an.differential[:4]))
    if an.documented_conditions:
        out.append("**Record review:** documented conditions — " + ", ".join(an.documented_conditions[:5]))
    if medications:
        out.append("**Record review:** medicines on record — " + "; ".join(m.split(" - ")[0] for m in medications[:4]))
    out += [f"⚠️ **Clinical inconsistency:** {i}" for i in an.inconsistencies[:2]]
    return "\n".join(f"- {o}" for o in out) if out else "- Nothing to analyse yet (no reports or symptoms)"


def risk_text(risk: RiskAssessment, is_patient: bool) -> str:
    indicators = "".join(f"\n- {i.indicator}" for i in risk.indicators[:5])
    if is_patient:
        text = f"Health risk from your symptoms and reports: **{LEVEL_BADGE[risk.clinical_level]}**"
        if any(i.category == "medication" for i in risk.indicators):
            text += " · ⛔ medicine safety alert"
        if risk.escalate:
            text += " · flagged for a clinician to check"
        return text + indicators
    text = f"Risk level **{LEVEL_BADGE[risk.risk_level]}** · urgency {risk.urgency.lower()}"
    if risk.escalate:
        text += " · **escalated for clinician validation**"
    return text + indicators


def risk_brief(risk: RiskAssessment) -> str:
    lines = [f"Risk level {risk.risk_level}, urgency {risk.urgency}, escalate={risk.escalate}"]
    return "\n".join(lines + [f"- [{i.level}] {i.indicator}" for i in risk.indicators[:10]])
