"""How a case is shown to each care-team role (RBAC: the radiologist sees imaging + contrast-safety alerts only)."""

from __future__ import annotations

from typing import Any, Callable, Dict

from clinicalchat.agents.specialists.ai import SpecialistReport
from clinicalchat.domain import workflow as wf
from clinicalchat.domain.risk import LEVEL_ICON
from clinicalchat.domain.roles import Role
from clinicalchat.presenters.findings import ALL_SPECIALISTS, specialists_text

Case = Dict[str, Any]


def _questions(case: Case) -> str:
    hist = case["payload"].get("history") or []
    if len(hist) <= 1:
        return f"**Patient asked:** {case['question']}  \n"
    return ("**Patient messages in this case:**\n"
            + "\n".join(f"{i}. {h['question']} _({h['at']})_" for i, h in enumerate(hist, 1)) + "\n\n")


def _head(case: Case, patient_label: str) -> str:
    return (f"### 🩺 Case {case['case_id']} — {patient_label}\n"
            f"**Stage:** {wf.stage_label(case)[0]} · **Risk:** {LEVEL_ICON.get(case['risk_level'], '')} {case['risk_level']}"
            + (" · escalated" if case["escalate"] else "") + "\n\n" + _questions(case))


def _radiology(pay: dict) -> str:
    reads = (pay.get("radiology") or {}).get("reads") or []
    return "".join(f"\n\n**🩻 Radiology read — {r['reader']}** ({r['at']}): _{r['verdict']}_"
                   + (f"\n> {r['report']}" if r.get("report") else "") for r in reads)


def _ai_specialists(pay: dict, show=ALL_SPECIALISTS) -> str:
    """The course AI specialists' opinions, limited to the record types the viewer's role may see."""
    data = pay.get("ai_specialists")
    if not data:
        return ""
    report = SpecialistReport.model_validate(data)
    shown = {k for k in show if getattr(report, k) is not None}
    if not shown:
        return ""
    return "\n\n**🧠 AI specialists (course agents)**\n" + specialists_text(report, shown)


def _triage(pay: dict) -> str:
    tri = pay.get("triage")
    if not tri:
        return ""
    return (f"\n\n**🧭 Triage — {tri['coordinator']}** ({tri['at']}): {wf.PRIORITY_ICON[tri['priority']]} **{tri['priority']}**, "
            f"assigned to **{tri['clinician_name']}**" + (f"\n> {tri['note']}" if tri.get("note") else ""))


def _timeline(pay: dict) -> str:
    events = pay.get("timeline") or []
    if not events:
        return ""
    return ("\n\n<details><summary>Case timeline</summary>\n\n"
            + "\n".join(f"- {e['at']} · **{e['actor']}** — {e['event']}" for e in events[-12:]) + "\n\n</details>")


def _imaging(pay: dict) -> str:
    md = ""
    for i, im in enumerate(pay.get("imaging") or [], 1):
        md += f"\n**🩻 Imaging {i}: {im['title']}**" + (f" · {im['date']}" if im.get("date") else "") + f" · _{im['source']}_"
        if im.get("impression"):
            md += f"\n- Impression (extracted): {im['impression']}"
        if im.get("red_flags"):
            md += "\n- 🚩 Red flags: " + ", ".join(im["red_flags"])
        if im.get("abnormal"):
            md += "\n- Abnormal: " + ", ".join(im["abnormal"])
        md += f"\n\n<details><summary>Report text</summary>\n\n```\n{im['text']}\n```\n</details>\n"
    return md


def radiologist_view(case: Case, patient_label: str) -> str:
    """Imaging + the contrast-safety alert only (no lab results, no allergy record, no full AI insight)."""
    pay = case["payload"]
    md = _head(case, patient_label) + _imaging(pay)
    alerts = pay.get("contrast_alerts") or []
    md += "\n\n**💉 Iodinated contrast check:** " + ("\n" + "\n".join(f"- {a}" for a in alerts) if alerts else "no allergy / interaction alert")
    md += _ai_specialists(pay, {"radiology"}) + _radiology(pay) + _timeline(pay)
    return md + "\n\n_Choose a verdict, write your read on the left and press **🩻 Submit read** — it goes to the care coordinator._"


def coordinator_view(case: Case, patient_label: str) -> str:
    pay, ins = case["payload"], case["payload"].get("insight", {})
    md = _head(case, patient_label)
    flags = [i for i in (pay.get("risk") or {}).get("indicators", []) if i.get("level") in ("HIGH", "CRITICAL")]
    if flags:
        md += "\n**🚨 Risk flags:**\n" + "\n".join(f"- {LEVEL_ICON.get(f['level'], '')} {f['indicator']}" for f in flags[:6])
    if pay.get("report_types"):
        md += "\n\n**📄 Reports in this case:** " + ", ".join(pay["report_types"])
    if pay.get("possible_conditions"):
        md += "\n\n**AI-suggested possibilities:** " + ", ".join(map(str, pay["possible_conditions"][:5]))
    if ins.get("next_steps"):
        md += "\n\n**Suggested next steps:** " + "; ".join(map(str, ins["next_steps"][:5]))
    md += _ai_specialists(pay) + _radiology(pay) + _timeline(pay)
    suggested = wf.suggested_priority(case["risk_level"])
    return md + (f"\n\n**Suggested priority:** {wf.PRIORITY_ICON[suggested]} {suggested} (from risk {case['risk_level']})"
                 "\n\n_Set the priority, pick a clinician on the left and press **🧭 Route to clinician**._")


def _insight(pay: dict) -> str:
    ins = pay.get("insight", {})
    if pay.get("insight_pending"):
        return "\n\n**💡 Clinical insight** — ⏳ the AI is writing it in the background; reopen the case in a moment."
    if ins.get("answer"):
        conf = ins.get("confidence")
        md = ("\n\n**💡 Clinical insight**" + (f" (confidence {conf:.0%})" if isinstance(conf, (int, float)) else "")
              + f"\n{ins['answer']}")
        return md + ("\n\n**Next steps:** " + "; ".join(map(str, ins["next_steps"])) if ins.get("next_steps") else "")
    if pay.get("insight_error"):
        return f"\n\n**💡 Clinical insight** — not available ({pay['insight_error'][:120]}). Use the findings above."
    return ""


def clinician_view(case: Case, patient_label: str) -> str:
    """Full findings, AI insight, radiology read, triage and the text shown to the patient."""
    pay = case["payload"]
    md = _head(case, patient_label)
    if pay.get("review_reasons"):
        md += "\n> 🟠 **Review Required** — " + "; ".join(pay["review_reasons"])
    if pay.get("facts"):
        md += "\n\n**🔬 Deterministic findings**\n```\n" + pay["facts"][:1500] + "\n```"
    md += _ai_specialists(pay, {"lab", "allergy"}) + _insight(pay) + _radiology(pay) + _triage(pay)
    md += ("\n\n---\n**💬 Shown to the patient as an AI assessment (latest):**\n\n"
           + "\n".join("> " + line for line in case["patient_draft"].splitlines()))
    if pay.get("guard_issues"):
        md += "\n\n**🛡️ Guard flags:** " + "; ".join(pay["guard_issues"])
    return md + _timeline(pay) + "\n\n_Add an optional note on the left, then **Approve & send** or **Return**._"


CASE_VIEW: Dict[Role, Callable[[Case, str], str]] = {
    Role.CLINICIAN: clinician_view, Role.RADIOLOGIST: radiologist_view, Role.COORDINATOR: coordinator_view}
