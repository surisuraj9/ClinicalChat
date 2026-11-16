"""HTML snippets of the page (header, patient card, case cards, queue, board, agent strip). Pure: data in, HTML out."""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.models import Domain, PatientProfile
from clinicalchat.domain.personas import Persona
from clinicalchat.domain.risk import LEVEL_ICON
from clinicalchat.domain.roles import AGENT_LABELS, AGENT_ORDER, AgentName, Role, policy
from clinicalchat.services.patients import SELF_REPORTED
from clinicalchat.web.content import AGENT_KIND, QUEUE_EMPTY, QUEUE_PICK, QUEUE_TITLE, STATE_TEXT
from clinicalchat.web.theme import LOGO_URL

STEP_ICON = {"done": "✓", "now": "●", "todo": "○", "skip": "–"}
NO_CASE = ("<div class='cc-muted'>No case yet. After you describe symptoms or attach reports, the AI assessment opens "
           "one case; the care team (radiologist for scans → care coordinator → clinician) then takes it forward.</div>")


def brand_html(llm_label: str) -> str:
    return (f"<div class='cc-brand'><img src='{LOGO_URL}' alt='logo'/><div><div class='t'>ClinicalChat</div>"
            f"<div class='s'>Multi-agent clinical assistant · {html.escape(llm_label)} · decision support only</div>"
            "</div></div>")


def user_html(persona: Persona) -> str:
    initials = "".join(w[0] for w in persona.display_name.replace("Dr. ", "").split()[:2]).upper()
    label = policy(persona.role).label
    return (f"<div class='cc-user'><div class='cc-avatar'>{initials}</div><div><div class='n'>{html.escape(persona.display_name)}</div>"
            f"<div class='r'><span class='cc-pill'>{label}</span> "
            f"{html.escape(persona.title) if persona.title != label else ''}</div></div></div>")


def patient_card(p: Optional[PatientProfile], domains: Sequence[Domain], symptoms: Sequence[str], can_upload: bool) -> str:
    if not p:
        return ("<div class='cc-muted'>No patient selected — questions search the whole record library. "
                "You can also name a patient in your question.</div>")
    meta = ", ".join(x for x in [f"{p.age} yrs" if p.age is not None else "", p.gender or ""] if x)
    tags = "".join(f"<span class='cc-tag'>{d.value.replace('_', ' ')}</span>" for d in domains)
    if not tags:
        tags = "<span class='cc-muted'>No reports on file yet" + (" — use 📎 Attach" if can_upload else "") + "</span>"
    if p.allergies:
        allergy = "<div class='cc-allergy'>⚠️ <b>Allergies:</b> " + html.escape(
            "; ".join(f"{a.allergen} ({'self-reported' if a.source == SELF_REPORTED else a.severity.lower()})"
                      for a in p.allergies)) + "</div>"
    else:
        allergy = "<div class='cc-ok'>✅ No documented allergies</div>"
    extra = []
    if symptoms:
        extra.append("🤒 " + html.escape(", ".join(symptoms[-6:])))
    if p.medications:
        extra.append(f"💊 {len(p.medications)} medication(s) on record")
    extra_html = "".join(f"<div class='cc-muted' style='margin-top:6px'>{x}</div>" for x in extra)
    return (f"<div class='cc-pname'>{html.escape(p.name)}</div><div class='cc-pmeta'>{meta}</div>"
            f"<div>{tags}</div>{allergy}{extra_html}")


def _stepper_html(case: dict) -> str:
    return "<div class='cc-steps'>" + "".join(
        f"<span class='{st}'>{STEP_ICON[st]} {html.escape(s)}</span>" for s, st in wf.stepper(case)) + "</div>"


def my_cases_html(cases: List[dict]) -> str:
    """The patient's case cards: open case first, then newest (at most 4)."""
    cases = [c for c in cases if c["status"] in wf.STAGE_LABEL]
    cases.sort(key=lambda c: c["created"], reverse=True)
    cases.sort(key=lambda c: c["status"] not in wf.OPEN_STATUSES)
    if not cases:
        return NO_CASE
    rows = []
    for c in cases[:4]:
        label, color = wf.stage_label(c)
        history = c["payload"].get("history") or []
        n = len(history or [None])
        tip = html.escape(" | ".join(h["question"] for h in history) or c["question"])
        last = (history or [{"question": c["question"]}])[-1]["question"] or "report upload"
        sub = f"{n} message{'s' if n > 1 else ''} · last: {html.escape(last)}"
        steps = _stepper_html(c) if c["status"] in wf.OPEN_STATUSES else ""
        rows.append(f"<div class='cc-case' title='{tip}'><div class='top'><b>{c['case_id']}</b>"
                    f"<span class='st' style='color:{color}'>{html.escape(label)}</span></div>{steps}<div class='sub'>{sub}</div></div>")
    return "".join(rows)


def review_title(n: int, role: Optional[Role] = Role.CLINICIAN) -> str:
    return f"<div class='cc-h'>{QUEUE_TITLE.get(role, 'Queue')}<span class='cc-badge'>{n}</span></div>"


def queue_choices(cases: List[dict], names: Dict[str, str], role: Optional[Role]) -> List[tuple]:
    """Dropdown entries of a work queue, with a placeholder first."""
    out = []
    for c in cases:
        pri = (c["payload"].get("triage") or {}).get("priority")
        tag = f"{pri} · " if pri else ""
        out.append((f"{LEVEL_ICON.get(c['risk_level'], '')} {tag}{names.get(c['patient_key'], c['patient_key'])} · "
                    f"{c['question'][:30]}", c["case_id"]))
    role = role or Role.CLINICIAN
    placeholder = QUEUE_PICK.get(role, "— Select a case —") if out else QUEUE_EMPTY.get(role, "✓ Nothing waiting")
    return [(placeholder, "")] + out


def board_html(cases: List[dict]) -> str:
    """Care coordinator's overview of every open case by stage."""
    counts = {s: 0 for s in wf.OPEN_STATUSES}
    urgent = 0
    for c in cases:
        if c["status"] in counts:
            counts[c["status"]] += 1
            urgent += c["risk_level"] in ("HIGH", "CRITICAL")
    cells = "".join(f"<div class='cc-bcell'><b>{counts[s]}</b><span>{wf.STAGE_LABEL[s][0]}</span></div>" for s in wf.OPEN_STATUSES)
    return (f"<div class='cc-board'>{cells}</div><div class='cc-muted' style='margin-top:4px'>⚠️ {urgent} open case(s) at "
            "high / critical risk</div>")


def agent_strip(role: Optional[Role], states: Optional[Dict[str, tuple]] = None) -> str:
    """Five equal tiles (the slide's pipeline) showing which agent is working on the current question."""
    states = states or {}
    allowed = policy(role).agents if role else set()
    tiles = []
    for a in AGENT_ORDER:
        state, note = states.get(a.value, ("idle" if a in allowed else "locked", ""))
        if role == Role.PATIENT and a == AgentName.INSIGHT and state == "idle":
            note = "for the care team"
        text = STATE_TEXT[state] + (f" · {note}" if note else "")
        tiles.append(f"<div class='cc-agent {state}' title='{html.escape(AGENT_LABELS[a] + ' — ' + text)}'>"
                     f"<div class='ak'>{AGENT_KIND[a]}</div><div class='an'>{AGENT_LABELS[a]}</div>"
                     f"<div class='as'>{html.escape(text)}</div></div>")
    return f"<div class='cc-agents'>{''.join(tiles)}</div>"
