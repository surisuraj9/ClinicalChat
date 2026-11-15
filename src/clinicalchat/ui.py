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
from clinicalchat.rbac import AGENT_LABELS, Role, policy

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
.gradio-container {max-width: 1320px !important; margin: 0 auto !important; background: #f6f8fb;}
.cc-top {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:10px 14px !important; align-items:center !important; gap:12px;}
.cc-brand {display:flex; align-items:center; gap:10px;}
.cc-brand img {width:38px; height:38px;}
.cc-brand .t {font-weight:700; font-size:19px; color:#111827; line-height:1.1;}
.cc-brand .s {font-size:12.5px; color:#6b7280;}
.cc-user {display:flex; align-items:center; justify-content:flex-end; gap:10px;}
.cc-avatar {width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#0d9488); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px;}
.cc-user .n {font-weight:600; color:#111827; font-size:14px; line-height:1.1;}
.cc-user .r {font-size:12px; color:#6b7280;}
.cc-pill {display:inline-block; padding:1px 8px; border-radius:999px; background:#eef2ff; color:#4338ca; font-size:11.5px; font-weight:600;}
.cc-card {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px !important;}
.cc-card h4 {margin:0 0 6px 0 !important; font-size:13px !important; color:#374151; text-transform:uppercase; letter-spacing:.04em;}
.cc-allergy {background:#fef2f2; border:1px solid #fecaca; color:#991b1b; border-radius:10px; padding:6px 10px; font-size:13px; margin-top:6px;}
.cc-ok {background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; border-radius:10px; padding:6px 10px; font-size:13px; margin-top:6px;}
.cc-tag {display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:999px; background:#f1f5f9; color:#334155; font-size:12px;}
.cc-chip button, button.cc-chip {justify-content:flex-start !important; text-align:left !important; font-weight:500 !important;
  background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; font-size:13px !important; white-space:normal !important;}
button.cc-chip:hover {background:#eef2ff !important; border-color:#c7d2fe !important;}
.cc-foot {font-size:12px; color:#6b7280; text-align:center;}
footer {display:none !important;}
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
            "<div class='s'>Lab · Radiology · Allergy · Care coordination agents</div></div></div>")


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
        return ("<div class='cc-card'><h4>Patient</h4><div style='font-size:13px;color:#6b7280'>No patient selected — "
                "questions will search the whole record library. You can also name a patient in your question.</div></div>")
    title = "Your record" if session.role == Role.PATIENT else "Selected patient"
    meta = ", ".join(x for x in [f"{p.age} yrs" if p.age is not None else "", p.gender or ""] if x)
    tags = "".join(f"<span class='cc-tag'>{d.value.replace('_', ' ')}</span>" for d in engine.patient_domains(p.patient_key))
    if p.allergies:
        allergy = "<div class='cc-allergy'>⚠️ <b>Allergies:</b> " + html.escape(
            "; ".join(f"{a.allergen} ({a.severity.lower()})" for a in p.allergies)) + "</div>"
    else:
        allergy = "<div class='cc-ok'>✅ No documented allergies</div>"
    meds = f"<div style='font-size:12.5px;color:#4b5563;margin-top:6px'>💊 {len(p.medications)} medication(s) on record</div>" if p.medications else ""
    return (f"<div class='cc-card'><h4>{title}</h4><div style='font-weight:600;font-size:15px;color:#111827'>{html.escape(p.name)}</div>"
            f"<div style='font-size:12.5px;color:#6b7280;margin-bottom:4px'>{meta}</div>{tags}{allergy}{meds}</div>")


def agents_card(role: Role) -> str:
    pol = policy(role)
    items = "".join(f"<div style='font-size:13px;margin:3px 0'>{AGENT_LABELS[a]}</div>" for a in sorted(pol.agents, key=lambda a: a.value))
    scope = "Your own record only" if pol.own_record_only else "All patients"
    return (f"<div class='cc-card'><h4>Your agents</h4>{items}"
            f"<div style='font-size:12px;color:#6b7280;margin-top:6px'>Data access: {scope}</div></div>")


def footer_html() -> str:
    info = describe_llm()
    llm = f"{info.get('provider')} · {info.get('model', '')}" if info.get("provider") != "unavailable" else "LLM unavailable"
    return (f"<div class='cc-foot'>LLM: {html.escape(llm)} · keyword routing (no LLM) · 1 LLM call per specialist · "
            "cached · every question is audit-logged · AI decision support — not a substitute for clinical judgement</div>")


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
    if not message:
        yield history, "", gr.update()
        return
    if session is None:
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": "Your session has not started yet — please reload the page."}]
        yield history, "", gr.update()
        return
    history.append({"role": "user", "content": message})
    log = ActivityLog()
    history.append(log.message())
    idx = len(history) - 1
    yield history, "", gr.update(interactive=False)
    try:
        for ev in get_orchestrator().ask(message, session):
            if ev.kind == "status":
                log.apply(ev)
                history[idx] = log.message()
            elif ev.kind == "safety":
                history.append({"role": "assistant", "content": f"### 🛡️ Safety check\n{ev.text}"})
            elif ev.kind == "blocked":
                history.append({"role": "assistant", "content": f"🔒 {ev.text}"})
            elif ev.kind == "answer":
                history.append({"role": "assistant", "content": f"### {ev.title}\n{ev.text}"})
            yield history, "", gr.update(interactive=False)
    except Exception as exc:  # never leave the UI hanging
        history.append({"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"})
    history[idx] = log.message(done=True)
    yield history, "", gr.update(interactive=True)


# ---------------------------------------------------------------------- app
def build_ui(auth_enabled: bool = True) -> gr.Blocks:
    with gr.Blocks(title="ClinicalChat", js=FORCE_LIGHT_JS) as demo:
        session = gr.State(None)

        with gr.Row(elem_classes="cc-top", equal_height=True):
            gr.HTML(brand_html(), scale=3)
            persona_pick = gr.Dropdown([(f"{p.display_name} — {policy(p.role).label}", p.username) for p in PERSONAS.values()],
                                       value="dr.sharma", label="Dev persona", visible=not auth_enabled, scale=2, container=False)
            user_box = gr.HTML(scale=3)
            gr.Button("Log out", link="/logout", size="sm", scale=0, min_width=90, visible=auth_enabled)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                patient_pick = gr.Dropdown(label="Patient", filterable=True, visible=False, interactive=True)
                card = gr.HTML()
                gr.Markdown("**Try asking**")
                chips = [gr.Button("", elem_classes="cc-chip", size="sm", visible=False) for _ in range(N_CHIPS)]
                agents_box = gr.HTML()
                with gr.Row():
                    export_btn = gr.Button("⬇️ Export summary", size="sm")
                    clear_btn = gr.Button("🧹 New chat", size="sm")
                export_file = gr.File(label="Chat summary (Markdown)", visible=False, interactive=False)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=620, show_label=False, render_markdown=True, group_consecutive_messages=False,
                                     placeholder="Ask about lab results, imaging, allergies or care status…")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your question and press Enter…", show_label=False, scale=8, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90)
        gr.HTML(footer_html())

        # ---------------- session bootstrap (identity comes from the authenticated username, server-side)
        def start(username: Optional[str]):
            persona = get_persona(username) or PERSONAS["dr.sharma"]
            s = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                            own_patient=persona.patient_key,
                            selected_patient=persona.patient_key or DEFAULT_PATIENT.get(persona.role))
            engine = get_engine()
            if persona.role == Role.PATIENT:
                pick = gr.update(visible=False, choices=[], value=None)
            else:
                choices = [(f"{p.label} · {', '.join(d.value.replace('_', ' ') for d in engine.patient_domains(p.patient_key))}", p.patient_key)
                           for p in engine.registry.all()]
                pick = gr.update(visible=True, choices=[("— No specific patient —", "")] + choices, value=s.selected_patient or "")
            ex = EXAMPLES[persona.role]
            chip_updates = [gr.update(value=ex[i], visible=True) if i < len(ex) else gr.update(visible=False) for i in range(N_CHIPS)]
            return [s, user_html(persona), pick, patient_card(s), agents_card(persona.role), welcome(persona),
                    gr.update(visible=False, value=None), *chip_updates]

        load_outputs = [session, user_box, patient_pick, card, agents_box, chatbot, export_file, *chips]

        def on_load(request: gr.Request):
            username = getattr(request, "username", None) if auth_enabled else None
            if not auth_enabled:
                username = "dr.sharma"
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

        submit_args = dict(fn=respond, inputs=[msg, chatbot, session], outputs=[chatbot, msg, send], concurrency_limit=4)
        msg.submit(**submit_args)
        send.click(**submit_args)
        for chip in chips:
            chip.click(fn=respond, inputs=[chip, chatbot, session], outputs=[chatbot, msg, send], concurrency_limit=4)

        def do_export(s: Optional[ChatSession]):
            if s is None:
                return gr.update(visible=False)
            return gr.update(value=str(export_session(s, get_engine())), visible=True)

        export_btn.click(do_export, session, export_file)

        def do_clear(s: Optional[ChatSession]):
            if s is not None:
                s.turns.clear()
                s.last_agents = []
            persona = get_persona(s.username) if s else PERSONAS["dr.sharma"]
            return welcome(persona), gr.update(visible=False, value=None)

        clear_btn.click(do_clear, session, [chatbot, export_file])
    return demo
