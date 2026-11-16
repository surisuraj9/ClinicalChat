"""The answers shown in the chat: the staff clinical insight, the patient's AI assessment and the safety banner."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from clinicalchat.agents.communication import bullets
from clinicalchat.agents.guard import blocking
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.domain.models import Domain, PatientProfile, SafetyAlert
from clinicalchat.domain.risk import LEVEL_BADGE
from clinicalchat.services.router import RoutePlan
from clinicalchat.services.safety import alert_line

AI_ASSESSMENT_NOTE = "<sub>🤖 AI assessment from your records — not a diagnosis. A clinician will confirm or correct it.</sub>"
AI_ANSWER_NOTE = ("<sub>🤖 AI answer from your records — not a diagnosis. No case was opened for this question; to have a "
                  "doctor check it, say *“please ask my doctor to review this”*.</sub>")


def badge(level: str) -> str:
    return LEVEL_BADGE.get(level, level)


def render_insight(res: AgentResult, review_reasons: List[str], sources: List[str]) -> str:
    if res.error:
        return f"_The Insight agent could not answer right now._ {res.error}"
    o = res.output or {}
    md = "> 🟠 **Review Required** — " + "; ".join(review_reasons) + "\n\n" if review_reasons else ""
    md += str(o.get("answer", "")).strip()
    md += (bullets("Key findings", o.get("key_findings")) + bullets("Possible conditions (to confirm)", o.get("possible_conditions"))
           + bullets("Recommended next steps", o.get("next_steps")) + bullets("Additional data needed", o.get("additional_data_needed")))
    conf = o.get("confidence")
    if isinstance(conf, (int, float)):
        md += f"\n\n**Confidence:** {conf:.0%}"
    srcs = list(dict.fromkeys([*map(str, o.get("sources") or []), *sources]))
    if srcs:
        md += f"\n\n<sub>Sources: {', '.join(srcs[:8])}</sub>"
    return md


def patient_assessment_markdown(insight: Dict[str, Any], risk_level: str, draft: str, issues: List[str],
                                reviewed: bool = True) -> str:
    """What the patient sees straight away: the AI's predicted health picture + plain-language explanation.
    Clearly labelled as an AI assessment awaiting clinician confirmation; unsafe wording is never shown."""
    conditions = [str(c) for c in insight.get("possible_conditions") or [] if str(c).strip()][:5]
    conf = insight.get("confidence")
    md = f"**Overall risk:** {badge(risk_level)}"
    if isinstance(conf, (int, float)) and not insight.get("ai_unavailable"):
        md += f" · **AI confidence:** {conf:.0%}"
    if conditions:
        who = "automatic checks" if insight.get("source") else "AI prediction"
        md += (f"\n\n**What your reports and symptoms may point to** _({who} — not a diagnosis)_:\n"
               + "\n".join(f"- {c}" for c in conditions))
    if blocking(issues):
        md += ("\n\n_The detailed explanation needs a clinician's check before I can show it (it contained wording "
               "that must be confirmed by a doctor). Please discuss these findings with your doctor._")
    elif draft.strip():
        md += "\n\n---\n" + draft
        if not any(w in draft.lower() for w in ("doctor", "clinician")):
            md += "\n\nPlease discuss these results with your doctor."
    return md + "\n\n" + (AI_ASSESSMENT_NOTE if reviewed else AI_ANSWER_NOTE)


def staff_draft_markdown(draft: str, issues: List[str]) -> str:
    """A patient-friendly draft written for a clinician, with the guard result."""
    if issues:
        return draft + "\n\n**🛡️ Guard check before sending:**\n" + "\n".join(f"- ⚠️ {i}" for i in issues)
    return draft + "\n\n<sub>🛡️ Guard check passed (no doses, no definitive diagnosis).</sub>"


def rule_based_findings(conditions: Sequence[str], findings: Sequence[str], level: str) -> tuple[str, str]:
    """When the AI is unavailable: (draft for the clinician, markdown for the patient) from the rule-based checks."""
    body = ""
    if findings:
        body += "\n\n**What the automatic checks found in your reports:**\n" + "\n".join(f"- {x}" for x in findings)
    if conditions:
        body += "\n\n**These findings can be seen with** _(not a diagnosis)_:\n" + "\n".join(f"- {c}" for c in conditions)
    risk_line = f"**Overall risk:** {badge(level)}"
    return risk_line + body + "\n\nPlease discuss these results with your clinician.", risk_line + " · _rule-based check (AI busy)_" + body


def general_answer_footer(sources: Sequence[str]) -> str:
    md = "\n\n<sub>ℹ️ General health information, not personal medical advice."
    return md + (f" Sources: {', '.join(sources)}</sub>" if sources else "</sub>")


def safety_banner(plan: RoutePlan, patient: Optional[PatientProfile], alerts: Sequence[SafetyAlert],
                  show_allergy_record: bool = True) -> str:
    """Deterministic medicine-safety lines shown before any AI text (never held back). Roles without access to
    allergy records (radiologist) still get the alerts for the medicine / contrast they asked about."""
    lines = list(plan.drug_notes)
    for a in alerts:
        icon = "⛔" if a.level == "CONTRAINDICATED" else "⚠️"
        lines.append(f"{icon} **{a.level}** — {alert_line(a).split('] ', 1)[1]}")
    if plan.drugs and not alerts:
        lines.append(f"✅ No allergy or interaction conflict found for **{', '.join(plan.drugs)}**"
                     + ("" if patient else " (select a patient to check their allergies)"))
    if show_allergy_record and patient and patient.allergies and (plan.drugs or plan.allergens or Domain.ALLERGY in plan.focus):
        lines.append("🩺 **Documented allergies:** " + "; ".join(
            f"{a.allergen} ({a.severity.lower()}{', ' + ', '.join(a.reactions[:2]) if a.reactions else ''})"
            for a in patient.allergies))
    return "\n".join(lines)
