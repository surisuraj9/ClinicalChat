"""Gradio chat UI: login per persona, patient context, example chips, live agent activity, export."""

from __future__ import annotations

import html
import os
import time
from typing import Dict, List, Optional

import gradio as gr

from clinicalchat.engine import get_engine
from clinicalchat.export import export_session
from clinicalchat.llm import describe_llm
from clinicalchat.orchestrator import ChatSession, Event, Orchestrator
from clinicalchat.personas import PERSONAS, Persona, get_persona
from clinicalchat.rbac import AGENT_LABELS, AgentName, Role, policy

LOGO_URL = "/brand/logo.svg"
N_CHIPS = 5

EXAMPLES: Dict[Role, List[str]] = {
    Role.PATIENT: [
        "Interpret my CBC results.",
        "Explain my thyroid results in simple terms.",
        "What follow-up tests are recommended based on these results?",
        "What are the cross-reactivities for this sulfa allergy?",
        "Is it safe for me to take ibuprofen?",
    ],
    Role.CLINICIAN: [
        "Analyze this patient's complete metabolic panel.",
        "Check for drug allergies before prescribing ibuprofen.",
        "What follow-up tests are recommended based on these results?",
        "What are the cross-reactivities for this sulfa allergy?",
        "Summarize this patient's care status for handoff.",
    ],
    Role.RADIOLOGIST: [
        "Review the chest CT findings.",
        "Give a differential diagnosis for this patient's imaging findings.",
        "Is iodinated contrast safe for this patient?",
        "What follow-up imaging is recommended?",
        "Review the chest X-ray findings.",
    ],
    Role.COORDINATOR: [
        "Summarize this patient's care status for handoff.",
        "What follow-up tests are recommended based on these results?",
        "Which specialists should be involved in this patient's care?",
        "Check for drug allergies before prescribing ibuprofen.",
        "What are the pending items for this patient?",
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
.cc-agents {display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px;}
.cc-agent {background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:9px 12px; height:62px; display:flex;
  flex-direction:column; justify-content:center; transition:all .2s; overflow:hidden;}
.cc-agent .an {font-weight:600; font-size:13.5px; color:#111827; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
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
@keyframes ccpulse {0%,100% {box-shadow:0 0 0 3px rgba(99,102,241,.12);} 50% {box-shadow:0 0 0 6px rgba(99,102,241,.22);}}

/* ---------- chat area ---------- */
.cc-chatcol {gap:10px !important; min-width:0 !important;}
#cc-chat {border:1px solid #e5e7eb !important; border-radius:14px !important; background:#fff !important;}
.cc-input {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:6px 8px !important; align-items:center !important;
  flex-wrap:nowrap !important;}
.cc-input .block {border:none !important; box-shadow:none !important;}
.cc-input {gap:8px !important;}
.cc-input button {height:42px !important; white-space:nowrap !important; width:116px !important; min-width:116px !important;
  flex:0 0 116px !important; font-size:14px !important; font-weight:600 !important; padding:0 10px !important; border-radius:10px !important;}
.cc-input button.cc-sec {background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; box-shadow:none !important;}
footer {display:none !important;}
@media (max-width: 900px) {.cc-main {flex-wrap:wrap !important;} .cc-side {height:auto;} .cc-agents {grid-template-columns:repeat(2, 1fr);}}
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
        f"<div style='font-size:12.5px;color:#6b7280;margin-top:8px'>Password for all demo accounts: {pwd}</div></div>"
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
    if p.allergies:
        allergy = "<div class='cc-allergy'>⚠️ <b>Allergies:</b> " + html.escape(
            "; ".join(f"{a.allergen} ({a.severity.lower()})" for a in p.allergies)) + "</div>"
    else:
        allergy = "<div class='cc-ok'>✅ No documented allergies</div>"
    meds = f"<div class='cc-muted' style='margin-top:8px'>💊 {len(p.medications)} medication(s) on record</div>" if p.medications else ""
    return (f"<div class='cc-pname'>{html.escape(p.name)}</div><div class='cc-pmeta'>{meta}</div>"
            f"<div>{tags}</div>{allergy}{meds}")


AGENT_ORDER = [AgentName.LAB, AgentName.RADIOLOGY, AgentName.ALLERGY, AgentName.COORDINATOR]
STATE_TEXT = {"idle": "Ready", "locked": "🔒 Not for your role", "queued": "Selected — waiting…",
              "working": "⏳ Working on your question…", "done": "✓ Answered", "failed": "✕ Could not answer",
              "denied": "🔒 Requested — not permitted for your role"}


def agent_strip(role: Optional[Role], states: Optional[Dict[str, tuple]] = None) -> str:
    """Four equal tiles showing which specialist agents handle the current question."""
    states = states or {}
    allowed = policy(role).agents if role else set()
    tiles = []
    for a in AGENT_ORDER:
        state, note = states.get(a.value, ("idle" if a in allowed else "locked", ""))
        text = STATE_TEXT[state] + (f" · {note}" if note else "")
        tiles.append(f"<div class='cc-agent {state}'><div class='an'>{AGENT_LABELS[a]}</div>"
                     f"<div class='as'>{html.escape(text)}</div></div>")
    return f"<div class='cc-agents'>{''.join(tiles)}</div>"


def llm_label() -> str:
    info = describe_llm()
    return f"{info.get('provider')} {info.get('model', '')}".strip() if info.get("provider") != "unavailable" else "LLM unavailable"


def welcome(persona: Persona) -> List[dict]:
    first = persona.display_name.split()[-1] if persona.role != Role.PATIENT else persona.display_name.split()[0]
    greet = f"Hello {('Dr. ' + first) if persona.display_name.startswith('Dr.') else first} 👋"
    if persona.role == Role.PATIENT:
        body = "I can explain your lab results, imaging reports and allergy safety in simple language. Try one of the suggestions on the left."
    else:
        body = "Pick a patient on the left (or name one in your question) and ask away. I'll route your question to the right specialist agents."
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


def respond(message: str, history: List[dict], session: Optional[ChatSession]):
    message = (message or "").strip()
    history = list(history or [])
    role = session.role if session else None
    if not message:
        yield history, "", gr.update(), gr.update()
        return
    if session is None:
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": "Your session has not started yet — please reload the page."}]
        yield history, "", gr.update(), gr.update()
        return
    history.append({"role": "user", "content": message})
    log = ActivityLog()
    history.append(log.message())
    idx = len(history) - 1
    states: Dict[str, tuple] = {}
    yield history, "", gr.update(interactive=False), agent_strip(role, states)
    try:
        for ev in get_orchestrator().ask(message, session):
            if ev.kind == "status":
                log.apply(ev)
                history[idx] = log.message()
                if ev.key == "route" and ev.status == "done":
                    plan = ev.data.get("plan", {})
                    for a in plan.get("agents", []):
                        states[AgentName(a).value] = ("queued", "")
                    for a in plan.get("denied", []):
                        states[AgentName(a).value] = ("denied", "")
                    if plan.get("agents"):
                        names = ", ".join(f"**{AGENT_LABELS[AgentName(a)]}**" for a in plan["agents"])
                        pt = get_engine().patient(plan.get("patient_key"))
                        history.append({"role": "assistant", "content":
                                        f"🧭 Routed to {names}" + (f" · Patient: **{pt.label}**" if pt else " · No specific patient")})
                elif ev.key.startswith("agent-"):
                    agent = ev.key.removeprefix("agent-")
                    if ev.status == "pending":
                        states[agent] = ("working", "")
                    else:
                        states[agent] = ("failed" if "failed" in ev.title else "done", ev.text)
            elif ev.kind == "safety":
                history.append({"role": "assistant", "content": f"### 🛡️ Safety check\n{ev.text}"})
            elif ev.kind == "blocked":
                history.append({"role": "assistant", "content": f"🔒 {ev.text}"})
            elif ev.kind == "answer":
                agent = ev.data.get("result", {}).get("agent")
                agent = getattr(agent, "value", agent)
                if agent and states.get(agent, ("",))[0] not in ("done", "failed"):
                    states[agent] = ("done", "no reports on file")
                history.append({"role": "assistant", "content": f"### {ev.title}\n{ev.text}"})
            yield history, "", gr.update(interactive=False), agent_strip(role, states)
    except Exception as exc:  # never leave the UI hanging
        history.append({"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"})
    history[idx] = log.message(done=True)
    yield history, "", gr.update(interactive=True), agent_strip(role, states)


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
            # ---------------- sidebar: three equal-width tiles
            with gr.Column(scale=0, min_width=340, elem_classes="cc-side"):
                with gr.Column(elem_classes="cc-tile"):
                    patient_title = gr.HTML("<div class='cc-h'>Patient</div>", padding=False)
                    patient_pick = gr.Dropdown(show_label=False, filterable=True, visible=False, interactive=True, container=False)
                    card = gr.HTML(padding=False)
                with gr.Column(elem_classes="cc-tile"):
                    gr.HTML("<div class='cc-h'>Try asking</div>", padding=False)
                    chips = [gr.Button("", elem_classes="cc-chip", size="sm", visible=False) for _ in range(N_CHIPS)]
            # ---------------- chat: agent strip + conversation + input
            with gr.Column(scale=1, elem_classes="cc-chatcol"):
                strip = gr.HTML(agent_strip(None), padding=False)
                chatbot = gr.Chatbot(elem_id="cc-chat", height="calc(100vh - 274px)", show_label=False, render_markdown=True,
                                     group_consecutive_messages=False,
                                     placeholder="Ask about lab results, imaging, allergies or care status…")
                with gr.Row(elem_classes="cc-input"):
                    msg = gr.Textbox(placeholder="Type your question and press Enter…", show_label=False, scale=8,
                                     autofocus=True, container=False)
                    send = gr.Button("Send ➤", variant="primary", scale=0, min_width=116)
                    export_btn = gr.DownloadButton("⬇️ Export", value=None, interactive=False, scale=0, min_width=116,
                                                   elem_classes="cc-sec")
                    clear_btn = gr.Button("🧹 New chat", scale=0, min_width=116, elem_classes="cc-sec")

        # ---------------- session bootstrap (identity comes from the authenticated username, server-side)
        def start(username: Optional[str]):
            persona = get_persona(username) or PERSONAS["dr.sharma"]
            s = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                            own_patient=persona.patient_key,
                            selected_patient=persona.patient_key or DEFAULT_PATIENT.get(persona.role))
            engine = get_engine()
            if persona.role == Role.PATIENT:
                pick = gr.update(visible=False, choices=[], value=None)
                title = "<div class='cc-h'>Your record</div>"
            else:
                choices = [(f"{p.label} · {', '.join(d.value.replace('_', ' ') for d in engine.patient_domains(p.patient_key))}", p.patient_key)
                           for p in engine.registry.all()]
                pick = gr.update(visible=True, choices=[("— No specific patient —", "")] + choices, value=s.selected_patient or "")
                title = "<div class='cc-h'>Patient</div>"
            ex = EXAMPLES[persona.role]
            chip_updates = [gr.update(value=ex[i], visible=True) if i < len(ex) else gr.update(visible=False) for i in range(N_CHIPS)]
            return [s, user_html(persona), title, pick, patient_card(s), agent_strip(persona.role), welcome(persona),
                    gr.update(value=None, interactive=False), *chip_updates]

        load_outputs = [session, user_box, patient_title, patient_pick, card, strip, chatbot, export_btn, *chips]

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

        outputs = [chatbot, msg, send, strip]
        def after_turn(s: Optional[ChatSession]):
            """Refresh the Export download and move the patient picker/card to the patient the question was about."""
            if s is None or not s.turns:
                return gr.update(value=None, interactive=False), gr.update(), gr.update()
            export = gr.update(value=str(export_session(s, get_engine())), interactive=True)
            if s.role != Role.PATIENT and s.last_patient and s.last_patient != s.selected_patient:
                s.selected_patient = s.last_patient
                return export, gr.update(value=s.last_patient), patient_card(s)
            return export, gr.update(), gr.update()

        post = [export_btn, patient_pick, card]

        msg.submit(respond, [msg, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)
        send.click(respond, [msg, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)
        for chip in chips:
            chip.click(respond, [chip, chatbot, session], outputs, concurrency_limit=4).then(after_turn, session, post)

        def do_clear(s: Optional[ChatSession]):
            if s is not None:
                s.turns.clear()
                s.last_agents = []
            persona = get_persona(s.username) if s else PERSONAS["dr.sharma"]
            return welcome(persona), gr.update(value=None, interactive=False), agent_strip(persona.role)

        clear_btn.click(do_clear, session, [chatbot, export_btn, strip])
    return demo
