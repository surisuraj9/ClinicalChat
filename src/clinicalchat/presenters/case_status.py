"""Deterministic answer to the patient's "what is happening with my case?" (no AI call)."""

from __future__ import annotations

from typing import Any, Dict, List

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.risk import LEVEL_BADGE

NO_CASE = ("You don't have a case yet. Describe how you feel or attach your reports with **📎 Attach** — the AI "
           "assessment opens a case for you and the care team takes it from there.")
_STEP = {"done": "✅ {}", "now": "**▶ {}**", "todo": "◻️ {}", "skip": "~~{}~~"}


def _current(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    open_ = [c for c in cases if c["status"] in wf.OPEN_STATUSES]
    return open_[0] if open_ else max(cases, key=lambda c: c["reviewed"] or c["created"])


def _care_team_lines(case: Dict[str, Any]) -> str:
    pay, md = case["payload"], ""
    conds = [str(c) for c in pay.get("possible_conditions", [])][:4]
    if conds:
        md += (f"**Latest AI assessment:** {LEVEL_BADGE.get(case['risk_level'], case['risk_level'])} — may point to "
               + ", ".join(conds) + " _(AI prediction — not a diagnosis)_\n\n")
    if pay.get("radiology"):
        urgent = any(r["verdict"] == wf.CRITICAL_VERDICT for r in pay["radiology"]["reads"])
        md += "🩻 **Radiologist:** your scans have been read" + (
            " — an urgent finding was flagged and the care team has been alerted." if urgent else ".") + "\n\n"
    tri = pay.get("triage") or {}
    if tri:
        md += (f"🧭 **Care coordinator:** priority {wf.PRIORITY_ICON.get(tri['priority'], '')} {tri['priority']}, "
               f"assigned to **{tri['clinician_name']}**.\n\n")
    return md


def _outcome(case: Dict[str, Any]) -> str:
    if case["status"] == wf.VALIDATED:
        return f"✅ **Confirmed by {case.get('reviewer') or 'your clinician'}:**\n\n{case.get('final_message') or ''}\n\n"
    if case["status"] == wf.RETURNED:
        return (f"↩️ **{case.get('reviewer') or 'Your clinician'} asked:** {case.get('review_note') or 'more information'} "
                "— reply here or attach a report; it goes into the same case.\n\n")
    tri = case["payload"].get("triage") or {}
    return "**Next:** " + wf.next_step_text(case["status"], tri.get("clinician_name")) + "\n\n"


def case_status_markdown(cases: List[Dict[str, Any]]) -> str:
    cases = [c for c in cases if c["status"] != wf.MERGED]
    if not cases:
        return NO_CASE
    case = _current(cases)
    md = f"**Case {case['case_id']}** — {wf.stage_label(case)[0]}\n\n"
    md += " → ".join(_STEP[st].format(s) for s, st in wf.stepper(case)) + "\n\n"
    md += _care_team_lines(case) + _outcome(case)
    events = case["payload"].get("timeline") or []
    if events:
        md += "<details><summary>Timeline</summary>\n\n" + "\n".join(
            f"- {e['at']} · **{e['actor']}** — {e['event']}" for e in events[-10:]) + "\n\n</details>"
    return md
