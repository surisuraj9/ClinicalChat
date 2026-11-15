"""Gradio chat UI: persona / self-registered login, 5-agent pipeline strip, live activity, report upload,
clinician review queue, patient case updates (polled) and export."""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import gradio as gr

from clinicalchat.engine import SELF_REPORTED, get_engine
from clinicalchat.export import export_session
from clinicalchat.llm import describe_llm
from clinicalchat.orchestrator import ChatSession, Event, Orchestrator
from clinicalchat.personas import PERSONAS, Persona, get_persona
from clinicalchat.extraction import SUPPORTED_UPLOADS
from clinicalchat.rbac import AGENT_LABELS, AGENT_ORDER, AgentName, Role, policy
from clinicalchat.store import get_store

LOGO_URL = "/brand/logo.svg"
N_CHIPS = 4
POLL_SECONDS = 10

EXAMPLES: Dict[Role, List[str]] = {
    Role.PATIENT: [
        "I have fever and cough since 3 days",
        "Explain my latest report in simple terms",
        "Is it safe for me to take ibuprofen?",
        "What is HbA1c?",
    ],
    Role.CLINICIAN: [
        "Interpret this patient's lab results.",
        "Check for drug allergies before prescribing ibuprofen.",
        "What follow-up tests are recommended based on these results?",
        "Explain this to the patient in simple terms.",
    ],
    Role.RADIOLOGIST: [
        "Review the chest CT findings.",
        "Is iodinated contrast safe for this patient?",
        "Give a differential diagnosis for the imaging findings.",
        "What follow-up imaging is recommended?",
    ],
    Role.COORDINATOR: [
        "Summarize this patient's care status for handoff.",
        "What follow-up tests are recommended based on these results?",
        "Which specialists should be involved in this patient's care?",
        "Check for drug allergies before prescribing ibuprofen.",
    ],
}
DEFAULT_PATIENT = {Role.CLINICIAN: "ananya-rao-54", Role.RADIOLOGIST: "meera-nair-47", Role.COORDINATOR: "john-doe-54"}

CSS = """
/* ---------- full-width, full-height layout ---------- */
.gradio-container {max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 12px 16px 4px !important;
  background: #f4f6fa !important;}
.gradio-container > .main, .gradio-container .wrap {max-width: 100% !important;}
.gradio-container > .main {padding: 0 !important;}
.gradio-container .contain {gap:12px !important;}
.cc-top {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:8px 16px !important; align-items:center !important;
  gap:12px; min-height:64px; flex-wrap:nowrap !important;}
.cc-brand {display:flex; align-items:center; gap:10px;}
.cc-brand img {width:38px; height:38px;}
.cc-brand .t {font-weight:700; font-size:19px; color:#111827; line-height:1.1;}
.cc-brand .s {font-size:12.5px; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-brand > div {min-width:0;}
.cc-user {display:flex; align-items:center; justify-content:flex-end; gap:10px;}
.cc-avatar {width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#0d9488); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; flex:none;}
.cc-user .n {font-weight:600; color:#111827; font-size:14px; line-height:1.15;}
.cc-user .r {font-size:12px; color:#6b7280;}
.cc-pill {display:inline-block; padding:1px 8px; border-radius:999px; background:#eef2ff; color:#4338ca; font-size:11.5px; font-weight:600;}
.cc-main {gap:16px !important; flex-wrap:nowrap !important;}

/* ---------- sidebar: identical full-width tiles ---------- */
.cc-side {flex: 0 0 340px !important; width:340px !important; max-width:340px !important; height: calc(100vh - 116px);
  overflow-y:auto; overflow-x:hidden; gap:12px !important; padding-right:2px;
  flex-direction:column !important; flex-wrap:nowrap !important;}
.cc-chatcol {flex: 1 1 auto !important;}
.cc-side > .cc-tile:last-child {flex:1 1 auto !important;}
.cc-tile {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:14px !important;
  padding:12px 14px !important; gap:6px !important; width:100% !important; box-shadow:0 1px 2px rgba(16,24,40,.04); flex:none !important;}
.cc-tile .cc-h {font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:.06em; margin:0;}
.cc-tile .block, .cc-tile .form {border:none !important; box-shadow:none !important; padding:0 !important; background:transparent !important;}
.cc-pname {font-weight:600; font-size:15px; color:#111827;}
.cc-pmeta {font-size:12.5px; color:#6b7280; margin:2px 0 6px;}
.cc-allergy, .cc-ok {border-radius:10px; padding:7px 10px; font-size:13px; margin-top:8px;}
.cc-allergy {background:#fef2f2; border:1px solid #fecaca; color:#991b1b;}
.cc-ok {background:#f0fdf4; border:1px solid #bbf7d0; color:#166534;}
.cc-tag {display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:999px; background:#f1f5f9; color:#334155; font-size:12px;}
.cc-muted {font-size:12.5px; color:#6b7280;}
button.cc-chip {justify-content:flex-start !important; text-align:left !important; font-weight:500 !important; width:100% !important;
  min-height:42px !important; height:42px !important; padding:6px 12px !important; line-height:1.3 !important;
  background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; font-size:13px !important;
  white-space:normal !important; border-radius:10px !important; box-shadow:none !important;}
button.cc-chip:hover {background:#eef2ff !important; border-color:#c7d2fe !important;}

/* ---------- agent strip: which agent is working on the question ---------- */
.cc-agents {display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px;}
.cc-agent {background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:8px 10px; height:62px; display:flex;
  flex-direction:column; justify-content:center; transition:all .2s; overflow:hidden;}
.cc-agent .an {font-weight:600; font-size:13px; color:#111827; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-agent .as {font-size:12px; color:#6b7280; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-agent.locked {background:#f8fafc; opacity:.6;}
.cc-agent.queued {border-color:#c7d2fe; background:#f5f7ff;}
.cc-agent.working {border-color:#6366f1; background:#eef2ff; box-shadow:0 0 0 3px rgba(99,102,241,.15); animation:ccpulse 1.4s infinite;}
.cc-agent.working .as {color:#4338ca; font-weight:600;}
.cc-agent.done {border-color:#86efac; background:#f0fdf4;}
.cc-agent.done .as {color:#166534; font-weight:600;}
.cc-agent.failed {border-color:#fca5a5; background:#fef2f2;}
.cc-agent.failed .as {color:#991b1b;}
.cc-agent.denied {border-color:#fcd34d; background:#fffbeb; opacity:1;}
.cc-agent.skipped {background:#f8fafc; opacity:.75;}
.cc-agent .ak {font-size:10.5px; color:#9ca3af; font-weight:600; letter-spacing:.04em;}
.cc-case {display:flex; justify-content:space-between; gap:8px; font-size:12.5px; padding:5px 0; border-bottom:1px dashed #eef0f4;}
.cc-case:last-child {border-bottom:none;}
.cc-case > span:first-child {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;}
.cc-case .st {white-space:nowrap; font-weight:600;}
.cc-badge {display:inline-block; min-width:20px; padding:0 7px; border-radius:999px; background:#4f46e5; color:#fff; font-size:11px;
  font-weight:700; text-align:center; margin-left:6px; text-transform:none; letter-spacing:0;}
.cc-rev-btns {gap:8px !important; flex-wrap:nowrap !important;}
.cc-rev-btns button {min-width:0 !important; flex:1 1 0 !important; width:auto !important;
  height:36px !important; font-size:13px !important; font-weight:600 !important; border-radius:10px !important;}
.cc-tile textarea, .cc-tile input {font-size:13px !important;}
@keyframes ccpulse {0%,100% {box-shadow:0 0 0 3px rgba(99,102,241,.12);} 50% {box-shadow:0 0 0 6px rgba(99,102,241,.22);}}

/* ---------- chat area ---------- */
.cc-chatcol {gap:10px !important; min-width:0 !important;}
#cc-chat {border:1px solid #e5e7eb !important; border-radius:14px !important; background:#fff !important;}
.cc-input {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:6px 8px !important; align-items:center !important;
  flex-wrap:nowrap !important;}
.cc-input .block {border:none !important; box-shadow:none !important;}
.cc-input {gap:8px !important;}
.cc-input button, .cc-input .cc-up button {height:42px !important; white-space:nowrap !important; width:116px !important; min-width:116px !important;
  flex:0 0 116px !important; font-size:14px !important; font-weight:600 !important; padding:0 10px !important; border-radius:10px !important;}
.cc-input button.cc-sec {background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; box-shadow:none !important;}
footer {display:none !important;}
@media (max-width: 900px) {.cc-main {flex-wrap:wrap !important;} .cc-side {height:auto;} .cc-agents {grid-template-columns:repeat(2, 1fr);}}
@media (max-width: 1280px) {.cc-input button {width:100px !important; min-width:100px !important; flex:0 0 100px !important;}}
"""

FORCE_LIGHT_JS = """
() => { const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }
"""


def login_message() -> str:
    rows = "".join(
        f"<tr><td style='padding:3px 10px;font-family:monospace'>{p.username}</td>"
        f"<td style='padding:3px 10px'>{html.escape(p.display_name)}</td>"
        f"<td style='padding:3px 10px;color:#4338ca'>{policy(p.role).label}</td></tr>"
        for p in PERSONAS.values()
    )
    pwd = "the demo password" if os.getenv("DEMO_PASSWORD") else "<b>demo@123</b>"
    return (
        "<div style='text-align:center;font-family:system-ui,sans-serif'>"
        f"<img src='{LOGO_URL}' alt='' style='width:56px;height:56px;margin:4px auto 6px;display:block'/>"
        "<div style='font-size:22px;font-weight:700;color:#111827'>ClinicalChat</div>"
        "<div style='font-size:13px;color:#6b7280;margin-bottom:10px'>Role-aware multi-agent clinical assistant</div>"
        "<table style='margin:0 auto;font-size:13px;border-collapse:collapse;text-align:left'>"
        f"<tr style='color:#6b7280'><th style='padding:3px 10px'>Username</th><th style='padding:3px 10px'>Name</th>"
        f"<th style='padding:3px 10px'>Role</th></tr>{rows}</table>"
        f"<div style='font-size:12.5px;color:#6b7280;margin-top:8px'>Password for all demo accounts: {pwd}</div>"
        "<div style='font-size:13.5px;margin-top:10px;padding:8px 10px;background:#eef2ff;border-radius:9px;color:#3730a3'>"
        "New patient? <a href='/signup' style='color:#4338ca;font-weight:700'>Create an account</a> to describe your "
        "symptoms and upload your reports.</div></div>"
    )


# ---------------------------------------------------------------------- lazy singletons
_ORCH: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        _ORCH = Orchestrator(get_engine())
    return _ORCH


# ---------------------------------------------------------------------- render helpers
def brand_html() -> str:
    return (f"<div class='cc-brand'><img src='{LOGO_URL}' alt='logo'/><div><div class='t'>ClinicalChat</div>"
            f"<div class='s'>Multi-agent clinical assistant · {html.escape(llm_label())} · decision support only</div>"
            "</div></div>")


def user_html(persona: Persona) -> str:
    initials = "".join(w[0] for w in persona.display_name.replace("Dr. ", "").split()[:2]).upper()
    return (f"<div class='cc-user'><div class='cc-avatar'>{initials}</div><div><div class='n'>{html.escape(persona.display_name)}</div>"
            f"<div class='r'><span class='cc-pill'>{policy(persona.role).label}</span> "
            f"{html.escape(persona.title) if persona.title != policy(persona.role).label else ''}</div></div></div>")


def patient_card(session: Optional[ChatSession]) -> str:
    if session is None:
        return ""
    engine = get_engine()
    key = session.own_patient or session.selected_patient
    p = engine.patient(key)
    if not p:
        return ("<div class='cc-muted'>No patient selected — questions search the whole record library. "
                "You can also name a patient in your question.</div>")
    meta = ", ".join(x for x in [f"{p.age} yrs" if p.age is not None else "", p.gender or ""] if x)
    tags = "".join(f"<span class='cc-tag'>{d.value.replace('_', ' ')}</span>" for d in engine.patient_domains(p.patient_key))
    if not tags:
        tags = "<span class='cc-muted'>No reports on file yet" + (" — use 📎 Upload" if policy(session.role).can_upload else "") + "</span>"
    if p.allergies:
        allergy = "<div class='cc-allergy'>⚠️ <b>Allergies:</b> " + html.escape(
            "; ".join(f"{a.allergen} ({'self-reported' if a.source == SELF_REPORTED else a.severity.lower()})"
                      for a in p.allergies)) + "</div>"
    else:
        allergy = "<div class='cc-ok'>✅ No documented allergies</div>"
    symptoms = engine.kg.patient_symptoms(p.patient_key)
    extra = []
    if symptoms:
        extra.append("🤒 " + html.escape(", ".join(symptoms[-6:])))
    if p.medications:
        extra.append(f"💊 {len(p.medications)} medication(s) on record")
    extra_html = "".join(f"<div class='cc-muted' style='margin-top:6px'>{x}</div>" for x in extra)
    return (f"<div class='cc-pname'>{html.escape(p.name)}</div><div class='cc-pmeta'>{meta}</div>"
            f"<div>{tags}</div>{allergy}{extra_html}")


CASE_STATUS = {"PENDING_REVIEW": ("⏳ Awaiting clinician", "#b45309"), "VALIDATED": ("✅ Reviewed", "#166534"),
               "RETURNED": ("↩️ More info needed", "#991b1b")}
RISK_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}


def my_cases_html(session: Optional[ChatSession]) -> str:
    if session is None or not session.own_patient:
        return ""
    cases = sorted(get_store().cases(patient_key=session.own_patient), key=lambda c: c["created"], reverse=True)[:4]
    if not cases:
        return "<div class='cc-muted'>No updates yet. When the agents prepare an explanation for you, a clinician checks it first — it will appear here and in the chat.</div>"
    rows = []
    for c in cases:
        label, color = CASE_STATUS[c["status"]]
        q = html.escape(c["question"][:42] + ("…" if len(c["question"]) > 42 else ""))
        rows.append(f"<div class='cc-case'><span title='{html.escape(c['question'])}'>{q}</span>"
                    f"<span class='st' style='color:{color}'>{label}</span></div>")
    return "".join(rows)


def review_choices() -> List[tuple]:
    engine = get_engine()
    out = []
    for c in get_store().cases(status="PENDING_REVIEW"):
        p = engine.patient(c["patient_key"])
        out.append((f"{RISK_ICON.get(c['risk_level'], '')} {p.name if p else c['patient_key']} · {c['question'][:34]}", c["case_id"]))
    placeholder = "— Select a case to review —" if out else "✓ No cases waiting"
    return [(placeholder, "")] + out


def review_title(n: int) -> str:
    return f"<div class='cc-h'>Review queue<span class='cc-badge'>{n}</span></div>"


def case_markdown(case: dict) -> str:
    engine = get_engine()
    p = engine.patient(case["patient_key"])
    pay = case["payload"]
    ins = pay.get("insight", {})
    md = (f"### 🩺 Case {case['case_id']} — {p.label if p else case['patient_key']}\n"
          f"**Patient asked:** {case['question']}  \n**Risk:** {RISK_ICON.get(case['risk_level'], '')} {case['risk_level']}"
          + (" · escalated" if case["escalate"] else ""))
    if pay.get("review_reasons"):
        md += "\n\n> 🟠 **Review Required** — " + "; ".join(pay["review_reasons"])
    if pay.get("facts"):
        md += "\n\n**🔬 Deterministic findings**\n```\n" + pay["facts"][:1500] + "\n```"
    if ins.get("answer"):
        md += f"\n\n**💡 Clinical insight** (confidence {ins.get('confidence', 0):.0%})\n{ins['answer']}"
        if ins.get("next_steps"):
            md += "\n\n**Next steps:** " + "; ".join(map(str, ins["next_steps"]))
    md += "\n\n---\n**💬 Draft for the patient:**\n\n" + "\n".join("> " + line for line in case["patient_draft"].splitlines())
    if pay.get("guard_issues"):
        md += "\n\n**🛡️ Guard flags:** " + "; ".join(pay["guard_issues"])
    md += "\n\n_Add an optional note on the left, then **Approve & send** or **Return**._"
    return md


STATE_TEXT = {"idle": "Ready", "locked": "🔒 Not for your role", "queued": "Queued…", "skipped": "Not needed",
              "working": "⏳ Working…", "done": "✓ Done", "failed": "✕ Failed", "denied": "🔒 Not permitted"}
AGENT_KIND = {AgentName.EXTRACTION: "STEP 1 · RULES", AgentName.ANALYSIS: "STEP 2 · RULES + KG", AgentName.RISK: "STEP 3 · RULES",
              AgentName.INSIGHT: "STEP 4 · AI", AgentName.COMMUNICATION: "STEP 5 · AI"}


def agent_strip(role: Optional[Role], states: Optional[Dict[str, tuple]] = None) -> str:
    """Five equal tiles (the slide's pipeline) showing which agent is working on the current question."""
    states = states or {}
    allowed = policy(role).agents if role else set()
    tiles = []
    for a in AGENT_ORDER:
        state, note = states.get(a.value, ("idle" if a in allowed else "locked", ""))
        if role == Role.PATIENT and a == AgentName.INSIGHT and state == "idle":
            note = "for your clinician"
        text = STATE_TEXT[state] + (f" · {note}" if note else "")
        tiles.append(f"<div class='cc-agent {state}' title='{html.escape(AGENT_LABELS[a] + ' — ' + text)}'>"
                     f"<div class='ak'>{AGENT_KIND[a]}</div><div class='an'>{AGENT_LABELS[a]}</div>"
                     f"<div class='as'>{html.escape(text)}</div></div>")
    return f"<div class='cc-agents'>{''.join(tiles)}</div>"


def llm_label() -> str:
    info = describe_llm()
    return f"{info.get('provider')} {info.get('model', '')}".strip() if info.get("provider") != "unavailable" else "LLM unavailable"


def welcome(persona: Persona) -> List[dict]:
    first = persona.display_name.split()[-1] if persona.role != Role.PATIENT else persona.display_name.split()[0]
    greet = f"Hello {('Dr. ' + first) if persona.display_name.startswith('Dr.') else first} 👋"
    if persona.role == Role.PATIENT:
        body = ("Tell me how you are feeling (e.g. *“I have fever and cough since 3 days”*), upload a report with **📎 Upload**, "
                "or ask any health question.\n\nFive agents work on your message: **📄 Data Extraction** reads your records, "
                "**🔬 Clinical Analysis** checks values, scans, allergies and symptoms, **🚨 Risk Detection** tells you right "
                "away if you need urgent care, **💡 Insight Generation** prepares a summary for your clinician and "
                "**💬 Patient Communication** writes a plain-language explanation that a clinician checks before you see it.")
    elif persona.role == Role.CLINICIAN:
        body = ("Pick a patient on the left (or name one in your question). Each question runs the 5-agent pipeline; "
                "patient cases waiting for your validation are in the **Review queue**.")
    else:
        body = "Pick a patient on the left (or name one in your question) and ask away — the agent strip shows who is working."
    return [{"role": "assistant", "content": f"{greet}\n\n{body}"}]


# ---------------------------------------------------------------------- chat streaming
class ActivityLog:
    """Single collapsible 'agent activity' message updated in place while agents work."""

    def __init__(self):
        self.steps: Dict[str, str] = {}
        self.started = time.time()
        self.current = "Starting…"

    def apply(self, ev: Event) -> None:
        icon = "⏳" if ev.status == "pending" else "✅"
        text = ev.text.replace("\n", " · ")
        self.steps[ev.key or ev.title] = f"{icon} **{ev.title}** — {text}" if text else f"{icon} **{ev.title}**"
        if ev.status == "pending":
            self.current = ev.title

    def message(self, done: bool = False) -> dict:
        elapsed = time.time() - self.started
        meta = {"title": "🧠 Agent activity" if done else f"⏳ {self.current}…", "status": "done" if done else "pending"}
        if done:
            meta["duration"] = round(elapsed, 1)
        return {"role": "assistant", "content": "\n\n".join(self.steps.values()) or "…", "metadata": meta}




def _stream(events: Iterator[Event], user_text: str, history: List[dict], session: ChatSession):
    """Render orchestrator events into the chat: one live activity log, one 'what the agents found' message
    (updated as each deterministic agent finishes), then safety / guidance / answers."""
    role = session.role
    history = list(history or [])
    history.append({"role": "user", "content": user_text})
    log = ActivityLog()
    history.append(log.message())
    idx = len(history) - 1
    findings: Dict[str, str] = {}
    f_idx: Optional[int] = None
    states: Dict[str, tuple] = {}
    session.busy = True
    yield history, agent_strip(role, states)
    try:
        for ev in events:
            if ev.kind == "status":
                log.apply(ev)
                history[idx] = log.message()
                if ev.key == "route" and ev.status == "done":
                    plan = ev.data.get("plan", {})
                    chosen = set(plan.get("agents", []))
                    for a in AGENT_ORDER:
                        if a in policy(role).agents:
                            states[a.value] = ("queued", "") if a.value in chosen else ("skipped", "")
                    if plan.get("denied_domains"):
                        states[AgentName.EXTRACTION.value] = ("denied" if not chosen else "queued",
                                                              "no " + ", ".join(plan["denied_domains"]) + " access")
                elif ev.key.startswith("agent-"):
                    agent = ev.key.removeprefix("agent-")
                    if ev.status == "pending":
                        states[agent] = ("working", "")
                    else:
                        short = ev.title.split(" — ", 1)[1] if " — " in ev.title else ""
                        failed = "failed" in ev.title
                        note = "" if short in ("done", "") else short
                        if agent == AgentName.INSIGHT.value and role == Role.PATIENT and not failed:
                            note = "sent to clinician"
                        states[agent] = ("failed" if failed else "done", note)
            elif ev.kind == "step":
                findings[ev.title] = f"**{ev.title}**\n{ev.text}"
                content = {"role": "assistant", "content": "### 🔎 What the agents found\n" + "\n\n".join(findings.values())}
                if f_idx is None:
                    history.append(content)
                    f_idx = len(history) - 1
                else:
                    history[f_idx] = content
            elif ev.kind == "safety":
                history.append({"role": "assistant", "content": f"### 🛡️ Safety check\n{ev.text}"})
            elif ev.kind == "guidance":
                history.append({"role": "assistant", "content": f"### 🚦 What to do now\n{ev.text}"})
            elif ev.kind == "blocked":
                history.append({"role": "assistant", "content": f"🔒 {ev.text}"})
            elif ev.kind == "case":
                history.append({"role": "assistant", "content": f"### {ev.title}\n{ev.text}"})
            elif ev.kind == "answer":
                history.append({"role": "assistant", "content": f"### {ev.title}\n{ev.text}"})
            yield history, agent_strip(role, states)
    except Exception as exc:  # never leave the UI hanging
        history.append({"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"})
    finally:
        session.busy = False
    history[idx] = log.message(done=True)
    yield history, agent_strip(role, states)


def respond(message: str, history: List[dict], session: Optional[ChatSession]):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        yield history, "", gr.update(), gr.update()
        return
    if session is None:
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": "Your session has not started yet — please reload the page."}]
        yield history, "", gr.update(), gr.update()
        return
    for h, strip in _stream(get_orchestrator().ask(message, session), message, history, session):
        yield h, "", gr.update(interactive=not session.busy), strip
    yield h, "", gr.update(interactive=True), strip


def respond_upload(file, history: List[dict], session: Optional[ChatSession]):
    history = list(history or [])
    if session is None or not file:
        yield history, gr.update(), gr.update()
        return
    path = Path(file if isinstance(file, str) else getattr(file, "name", ""))
    for h, strip in _stream(get_orchestrator().upload(path, session, display_name=path.name), f"📎 Uploaded **{path.name}**",
                            history, session):
        yield h, gr.update(interactive=not session.busy), strip
    yield h, gr.update(interactive=True), strip


def deliver_updates(session: Optional[ChatSession], history: List[dict]) -> Optional[List[dict]]:
    """Validated / returned cases for a patient -> chat messages (once). None = nothing new."""
    if session is None or not session.own_patient or session.busy:
        return None
    store = get_store()
    new = store.undelivered(session.own_patient)
    if not new:
        return None
    history = list(history or [])
    for c in new:
        who = c.get("reviewer") or "your clinician"
        if c["status"] == "VALIDATED":
            history.append({"role": "assistant", "content":
                            f"### ✅ Reviewed by {who}\n_Your question: “{c['question']}”_\n\n{c['final_message']}"})
        else:
            note = c.get("review_note") or "Your clinician needs a little more information."
            history.append({"role": "assistant", "content":
                            f"### ↩️ {who} needs more information\n_Your question: “{c['question']}”_\n\n{note}\n\n"
                            "You can reply here (for example describe your symptoms in more detail) or upload a report with 📎."})
        store.mark_delivered(c["case_id"])
    return history


def staff_patient_choices() -> List[tuple]:
    engine = get_engine()
    return [("— No specific patient —", "")] + [
        (f"{p.label} · {', '.join(d.value.replace('_', ' ') for d in engine.patient_domains(p.patient_key)) or 'new patient'}",
         p.patient_key) for p in engine.registry.all()]


# ---------------------------------------------------------------------- app
def build_ui(auth_enabled: bool = True) -> gr.Blocks:
    with gr.Blocks(title="ClinicalChat", js=FORCE_LIGHT_JS, fill_width=True) as demo:
        session = gr.State(None)

        with gr.Row(elem_classes="cc-top", equal_height=True):
            gr.HTML(brand_html(), scale=3, padding=False)
            persona_pick = gr.Dropdown([(f"{p.display_name} — {policy(p.role).label}", p.username) for p in PERSONAS.values()],
                                       value="dr.sharma", label="Dev persona", visible=not auth_enabled, scale=2, container=False)
            user_box = gr.HTML(scale=3, padding=False)
            gr.Button("Log out", link="/logout", size="sm", scale=0, min_width=90, visible=auth_enabled)

        with gr.Row(elem_classes="cc-main", equal_height=False):
            # ---------------- sidebar tiles (same width, stacked)
            with gr.Column(scale=0, min_width=340, elem_classes="cc-side"):
                with gr.Column(elem_classes="cc-tile"):
                    patient_title = gr.HTML("<div class='cc-h'>Patient</div>", padding=False)
                    patient_pick = gr.Dropdown(show_label=False, filterable=True, visible=False, interactive=True, container=False)
                    card = gr.HTML(padding=False)
                with gr.Column(elem_classes="cc-tile", visible=False) as review_tile:
                    rev_title = gr.HTML(review_title(0), padding=False)
                    review_pick = gr.Dropdown(show_label=False, choices=[], value=None, interactive=True, container=False,
                                              filterable=False)
                    review_note = gr.Textbox(show_label=False, placeholder="Note to patient / feedback for the agents (optional)",
                                             lines=2, max_lines=3, container=False)
                    with gr.Row(elem_classes="cc-rev-btns"):
                        approve_btn = gr.Button("✅ Approve & send", variant="primary", size="sm")
                        return_btn = gr.Button("↩️ Return", size="sm", elem_classes="cc-sec")
                with gr.Column(elem_classes="cc-tile", visible=False) as cases_tile:
                    gr.HTML("<div class='cc-h'>My case updates</div>", padding=False)
                    cases_box = gr.HTML(padding=False)
                with gr.Column(elem_classes="cc-tile"):
                    gr.HTML("<div class='cc-h'>Try asking</div>", padding=False)
                    chips = [gr.Button("", elem_classes="cc-chip", size="sm", visible=False) for _ in range(N_CHIPS)]
            # ---------------- chat: agent strip + conversation + input
            with gr.Column(scale=1, elem_classes="cc-chatcol"):
                strip = gr.HTML(agent_strip(None), padding=False)
                chatbot = gr.Chatbot(elem_id="cc-chat", height="calc(100vh - 274px)", show_label=False, render_markdown=True,
                                     group_consecutive_messages=False,
                                     placeholder="Describe your symptoms, upload a report or ask a health question…")
                with gr.Row(elem_classes="cc-input"):
                    msg = gr.Textbox(placeholder="Type your question and press Enter…", show_label=False, scale=8,
                                     autofocus=True, container=False)
                    send = gr.Button("Send ➤", variant="primary", scale=0, min_width=116)
                    upload_btn = gr.UploadButton("📎 Upload", file_types=list(SUPPORTED_UPLOADS), file_count="single",
                                                 scale=0, min_width=116, elem_classes="cc-sec", visible=False)
                    export_btn = gr.DownloadButton("⬇️ Export", value=None, interactive=False, scale=0, min_width=116,
                                                   elem_classes="cc-sec")
                    clear_btn = gr.Button("🧹 New chat", scale=0, min_width=116, elem_classes="cc-sec")
        timer = gr.Timer(POLL_SECONDS)

        # ---------------- session bootstrap (identity comes from the authenticated username, server-side)
        def start(username: Optional[str]):
            persona = get_persona(username) or PERSONAS["dr.sharma"]
            pol = policy(persona.role)
            s = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                            own_patient=persona.patient_key,
                            selected_patient=persona.patient_key or DEFAULT_PATIENT.get(persona.role))
            if persona.role == Role.PATIENT:
                pick = gr.update(visible=False, choices=[], value=None)
                title = "<div class='cc-h'>Your record</div>"
            else:
                pick = gr.update(visible=True, choices=staff_patient_choices(), value=s.selected_patient or "")
                title = "<div class='cc-h'>Patient</div>"
            ex = EXAMPLES[persona.role]
            chip_updates = [gr.update(value=ex[i], visible=True) if i < len(ex) else gr.update(visible=False) for i in range(N_CHIPS)]
            history = welcome(persona)
            history = deliver_updates(s, history) or history
            pending = review_choices() if pol.can_review else [("", "")]
            return [s, user_html(persona), title, pick, patient_card(s), agent_strip(persona.role), history,
                    gr.update(value=None, interactive=False), gr.update(visible=pol.can_upload),
                    gr.update(visible=pol.can_review), review_title(len(pending) - 1), gr.update(choices=pending, value=""),
                    gr.update(visible=persona.role == Role.PATIENT), my_cases_html(s), *chip_updates]

        load_outputs = [session, user_box, patient_title, patient_pick, card, strip, chatbot, export_btn, upload_btn,
                        review_tile, rev_title, review_pick, cases_tile, cases_box, *chips]

        def on_load(request: gr.Request):
            username = getattr(request, "username", None) if auth_enabled else "dr.sharma"
            return start(username)

        demo.load(on_load, outputs=load_outputs)
        if not auth_enabled:
            persona_pick.change(start, inputs=persona_pick, outputs=load_outputs)

        def pick_patient(key: str, s: Optional[ChatSession]):
            if s is None or s.role == Role.PATIENT:  # patients are locked to their own record
                return s, patient_card(s)
            s.selected_patient = key or None
            return s, patient_card(s)

        patient_pick.change(pick_patient, [patient_pick, session], [session, card])

        def after_turn(s: Optional[ChatSession]):
            """Refresh export, sidebar (record / cases / review queue) and move the picker to the question's patient."""
            if s is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            export = gr.update(value=str(export_session(s, get_engine())), interactive=True) if s.turns else gr.update()
            pick = gr.update()
            if s.role != Role.PATIENT and s.last_patient and s.last_patient != s.selected_patient:
                s.selected_patient = s.last_patient
                pick = gr.update(value=s.last_patient)
            if policy(s.role).can_review:
                pending = review_choices()
                rev = (review_title(len(pending) - 1), gr.update(choices=pending))
            else:
                rev = (gr.update(), gr.update())
            return export, pick, patient_card(s), my_cases_html(s), *rev

        post = [export_btn, patient_pick, card, cases_box, rev_title, review_pick]
        outputs = [chatbot, msg, send, strip]
        msg.submit(respond, [msg, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)
        send.click(respond, [msg, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)
        for chip in chips:
            chip.click(respond, [chip, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)
        upload_btn.upload(respond_upload, [upload_btn, chatbot, session], [chatbot, send, strip],
                          concurrency_limit=2).then(after_turn, session, post)

        # ---------------- clinician review queue
        def open_case(case_id: Optional[str], history: List[dict], s: Optional[ChatSession]):
            nothing = (gr.update(), gr.update(), gr.update())
            if not case_id or s is None or not policy(s.role).can_review:
                return nothing
            case = get_store().case(case_id)
            if not case:
                return nothing
            s.selected_patient = case["patient_key"]
            pick = gr.update(choices=staff_patient_choices(), value=case["patient_key"])
            return list(history or []) + [{"role": "assistant", "content": case_markdown(case)}], pick, patient_card(s)

        review_pick.change(open_case, [review_pick, chatbot, session], [chatbot, patient_pick, card])

        def decide(approve: bool):
            def _do(case_id: Optional[str], note: str, history: List[dict], s: Optional[ChatSession]):
                history = list(history or [])
                if s is None or not policy(s.role).can_review:
                    return history, gr.update(), gr.update(), gr.update()
                if not case_id:
                    history.append({"role": "assistant", "content": "Select a case from the **Review queue** first."})
                    return history, gr.update(), gr.update(), gr.update()
                case = get_orchestrator().review(case_id, s.display_name, approve, note)
                p = get_engine().patient(case["patient_key"])
                verb = "approved and sent to" if approve else "returned to"
                history.append({"role": "assistant", "content":
                                f"{'✅' if approve else '↩️'} Case **{case_id}** {verb} **{p.name if p else case['patient_key']}**."
                                + (f"\n\n_Your note will also help the Insight agent on future cases:_ “{note.strip()}”" if note.strip() else "")})
                pending = review_choices()
                return history, review_title(len(pending) - 1), gr.update(choices=pending, value=""), ""
            return _do

        rev_outputs = [chatbot, rev_title, review_pick, review_note]
        approve_btn.click(decide(True), [review_pick, review_note, chatbot, session], rev_outputs)
        return_btn.click(decide(False), [review_pick, review_note, chatbot, session], rev_outputs)

        # ---------------- polling: patients receive reviewed answers, clinicians see new cases
        def poll(s: Optional[ChatSession], history: List[dict]):
            if s is None:
                return gr.update(), gr.update(), gr.update(), gr.update()
            if s.role == Role.PATIENT:
                updated = deliver_updates(s, history)
                if updated is None:
                    return gr.update(), gr.update(), gr.update(), gr.update()
                return updated, my_cases_html(s), gr.update(), gr.update()
            if policy(s.role).can_review:
                pending = review_choices()
                return gr.update(), gr.update(), review_title(len(pending) - 1), gr.update(choices=pending)
            return gr.update(), gr.update(), gr.update(), gr.update()

        timer.tick(poll, [session, chatbot], [chatbot, cases_box, rev_title, review_pick], show_progress="hidden")

        def do_clear(s: Optional[ChatSession]):
            if s is not None:
                s.turns.clear()
                s.last_insight = None
            persona = get_persona(s.username) if s else PERSONAS["dr.sharma"]
            return welcome(persona), gr.update(value=None, interactive=False), agent_strip(persona.role)

        clear_btn.click(do_clear, session, [chatbot, export_btn, strip])
    return demo
