"""Gradio event handlers: each UI action calls ONE service of the container and returns component updates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import gradio as gr

from clinicalchat.container import Container
from clinicalchat.domain import workflow as wf
from clinicalchat.domain.conversation import ChatSession
from clinicalchat.domain.personas import PERSONAS, Persona
from clinicalchat.domain.roles import Role, policy
from clinicalchat.ingestion.uploads import MAX_UPLOAD_FILES
from clinicalchat.presenters.allergy import allergy_card_markdown
from clinicalchat.web import components, content
from clinicalchat.web.chat_renderer import render
from clinicalchat.web.sidebar import Sidebar, has_queue
from clinicalchat.presenters.case_views import CASE_VIEW

History = List[dict]
DEFAULT_USER = "dr.sharma"
SKIP = gr.update()


def _say(history: History, text: str) -> History:
    return list(history or []) + [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------------------- attachments (📎 chips)
def _upload_paths(files) -> List[Path]:
    items = files if isinstance(files, (list, tuple)) else [files] if files else []
    names = (f if isinstance(f, str) else getattr(f, "path", None) or getattr(f, "name", "") for f in items)
    return [Path(n) for n in names if n]


def attach_label(paths: List[str]) -> dict:
    """One removable chip per attached file (click × on a chip to drop just that file)."""
    if not paths:
        return gr.update(visible=False, choices=[], value=[])
    return gr.update(visible=True, choices=[(f"📎 {Path(p).name}", p) for p in paths], value=list(paths))


def attach_placeholder(paths: List[str]) -> dict:
    return gr.update(placeholder=content.ATTACHED_PLACEHOLDER if paths else content.DEFAULT_PLACEHOLDER)


def attach_files(files, attached: List[str], _session: Optional[ChatSession] = None):
    """📎 only attaches; nothing is analysed until the user presses Send (with or without a message)."""
    current = list(attached or [])
    names = {Path(p).name for p in current}
    for p in _upload_paths(files):
        if p.name not in names and len(current) < MAX_UPLOAD_FILES:
            current.append(str(p))
            names.add(p.name)
    return current, attach_label(current), attach_placeholder(current)


def clear_attachments():
    return [], attach_label([]), attach_placeholder([])


def update_attachments(selected: Optional[List[str]], attached: List[str]):
    """A chip was removed: keep only the files still selected (order preserved)."""
    keep = [p for p in (attached or []) if p in set(selected or [])]
    if keep == list(attached or []):
        return SKIP, SKIP, SKIP
    return keep, attach_label(keep), attach_placeholder(keep)


class UIController:
    def __init__(self, container: Container):
        self.c = container
        self.sidebar = Sidebar(container)
        self._shown_cases: Dict[str, str] = {}  # session_id -> last case-cards HTML (poll only redraws on change)

    def persona(self, username: Optional[str]) -> Persona:
        return self.c.accounts.get_persona(username) or PERSONAS[DEFAULT_USER]

    # ------------------------------------------------------------------------ session bootstrap
    def start(self, username: Optional[str]) -> list:
        """Identity comes from the authenticated username (server-side); the browser cannot pick its role."""
        persona = self.persona(username)
        role = persona.role
        s = ChatSession(username=persona.username, role=role, display_name=persona.display_name,
                        own_patient=persona.patient_key,
                        selected_patient=persona.patient_key or content.DEFAULT_PATIENT.get(role))
        if role == Role.PATIENT:
            pick, title = gr.update(visible=False, choices=[], value=None), "<div class='cc-h'>Your record</div>"
        else:
            pick = gr.update(visible=True, choices=self.sidebar.staff_patient_choices(), value=s.selected_patient or "")
            title = "<div class='cc-h'>Patient</div>"
        ex = content.EXAMPLES[role]
        chips = [gr.update(value=ex[i], visible=True) if i < len(ex) else gr.update(visible=False)
                 for i in range(content.N_CHIPS)]
        greeting = content.welcome(persona)
        history = self.deliver_updates(s, greeting) or greeting
        queue = has_queue(role)
        pending = self.sidebar.queue_choices(s) if queue else [("", "")]
        return [s, components.user_html(persona), title, pick, self.sidebar.patient_card(s), components.agent_strip(role),
                history, gr.update(value=None, interactive=False), gr.update(visible=policy(role).can_upload),
                gr.update(visible=queue), components.review_title(len(pending) - 1, role), gr.update(choices=pending, value=""),
                gr.update(visible=role == Role.PATIENT), self.sidebar.my_cases(s), *chips,
                gr.update(visible=role == Role.COORDINATOR, value=self.sidebar.board() if role == Role.COORDINATOR else ""),
                gr.update(visible=role == Role.RADIOLOGIST), gr.update(visible=role == Role.COORDINATOR),
                gr.update(placeholder=content.NOTE_PLACEHOLDER.get(role, ""), value=""),
                gr.update(value=content.ACTION_LABEL.get(role, "Submit")), gr.update(visible=role == Role.CLINICIAN)]

    def pick_patient(self, key: str, s: Optional[ChatSession]):
        if s is not None and s.role != Role.PATIENT:  # patients are locked to their own record
            s.selected_patient = key or None
        return s, self.sidebar.patient_card(s)

    # ------------------------------------------------------------------------ chat
    def respond(self, message: str, history: History, session: Optional[ChatSession]):
        message = (message or "").strip()
        history = list(history or [])
        if not message:
            yield history, "", SKIP, SKIP
            return
        if session is None:
            history.append({"role": "user", "content": message})
            yield _say(history, "Your session has not started yet — please reload the page."), "", SKIP, SKIP
            return
        yield from self._stream(render(self.c.chat.ask(message, session), message, history, session), session)

    def respond_with_files(self, message: str, history: History, session: Optional[ChatSession], attached: List[str]):
        """Send button / Enter: the message plus any attached reports go through ONE pipeline run."""
        paths = [Path(p) for p in (attached or [])]
        if not paths:
            for out in self.respond(message, history, session):
                yield (*out, SKIP, SKIP, SKIP)
            return
        if session is None:
            yield list(history or []), "", SKIP, SKIP, SKIP, SKIP, SKIP
            return
        message = (message or "").strip()
        files_md = "\n".join(f"- 📎 {p.name}" for p in paths)
        label = f"{message}\n\n{files_md}" if message else f"📎 **{len(paths)} report(s) attached**\n{files_md}"
        events = self.c.uploads.upload_many([(p, p.name) for p in paths], session, message=message)
        cleared = clear_attachments()
        for out in self._stream(render(events, label, history, session), session):
            yield (*out, *cleared)

    @staticmethod
    def _stream(frames: Iterator, session: ChatSession):
        h = strip = None
        for h, strip in frames:
            yield h, "", gr.update(interactive=not session.busy), strip
        yield h, "", gr.update(interactive=True), strip

    def after_turn(self, s: Optional[ChatSession]):
        """Refresh export, sidebar (record / cases / queue) and move the picker to the question's patient."""
        if s is None:
            return (SKIP,) * 6
        export = gr.update(value=str(self.c.exporter.export(s)), interactive=True) if s.turns else SKIP
        pick = SKIP
        if s.role != Role.PATIENT and s.last_patient and s.last_patient != s.selected_patient:
            s.selected_patient = s.last_patient
            pick = gr.update(value=s.last_patient)
        return (export, pick, self.sidebar.patient_card(s), self.sidebar.my_cases(s), *self._queue_updates(s))

    def allergy_check(self, allergen: str, history: History, s: Optional[ChatSession]):
        """🛡️ Allergy Safety Check tile: knowledge-graph cross-reactivity card posted into the chat (no AI)."""
        allergen = (allergen or "").strip()
        history = list(history or [])
        if not allergen or s is None:
            return history, ""
        kg = self.c.kb.kg
        trace = self.c.tracer.trace("allergy-safety-check", user_id=s.username, session_id=s.session_id,
                                    tags=["allergy-check", s.role.value], input={"allergen": allergen})
        with trace.step("knowledge-graph", as_type="tool", input={"allergen": allergen}) as span:
            card = kg.allergy_safety_card(allergen)
            span.update(output=card)
        trace.end({"allergen": card["allergen"], "cross_reactive": len(card["cross_reactive_allergens"])})
        history.append({"role": "user", "content": f"🛡️ Allergy safety check: **{allergen}**"})
        history.append({"role": "assistant", "content": allergy_card_markdown(card, kg.known_allergens())})
        return history, ""

    def clear(self, s: Optional[ChatSession]):
        if s is not None:
            s.turns.clear()
            s.last_insight = None
        persona = self.persona(s.username) if s else PERSONAS[DEFAULT_USER]
        return content.welcome(persona), gr.update(value=None, interactive=False), components.agent_strip(persona.role)

    # ------------------------------------------------------------------------ care-team work queues
    def _queue_updates(self, s: ChatSession) -> tuple:
        if not has_queue(s.role):
            return SKIP, SKIP
        pending = self.sidebar.queue_choices(s)
        return components.review_title(len(pending) - 1, s.role), gr.update(choices=pending)

    def _board_update(self, s: ChatSession):
        return gr.update(value=self.sidebar.board()) if s.role == Role.COORDINATOR else SKIP

    def open_case(self, case_id: Optional[str], history: History, s: Optional[ChatSession]):
        nothing = (SKIP,) * 5
        if not case_id or s is None or not has_queue(s.role):
            return nothing
        case = self.c.care_team.case(case_id)
        if not case or case["status"] != wf.QUEUE_FOR_ROLE[s.role]:
            return nothing
        s.selected_patient = case["patient_key"]
        p = self.c.kb.patients.get(case["patient_key"])
        view = CASE_VIEW[s.role](case, p.label if p else case["patient_key"])
        pick = gr.update(choices=self.sidebar.staff_patient_choices(), value=case["patient_key"])
        priority = gr.update(value=wf.suggested_priority(case["risk_level"])) if s.role == Role.COORDINATOR else SKIP
        verdict = gr.update(value=wf.RADIOLOGY_VERDICTS[0]) if s.role == Role.RADIOLOGIST else SKIP
        return _say(history, view), pick, self.sidebar.patient_card(s), priority, verdict

    def decide(self, approve: bool):
        """Handler for the queue's main button (approve / submit read / route) or ↩️ Return."""
        def _do(case_id: Optional[str], note: str, verdict: str, priority: str, assignee: str, history: History,
                s: Optional[ChatSession]):
            keep = (SKIP,) * 4
            if s is None or not has_queue(s.role):
                return list(history or []), *keep
            if not case_id:
                return _say(history, f"Select a case from the **{content.QUEUE_TITLE[s.role]}** first."), *keep
            note = (note or "").strip()
            if s.role == Role.RADIOLOGIST and verdict != wf.RADIOLOGY_VERDICTS[0] and not note:
                return _say(history, "Please write your findings / impression in the box before submitting a "
                                     "**different** or **critical** read."), *keep
            try:
                msg = self._act(s, case_id, approve, note, verdict, priority, assignee)
            except ValueError as exc:
                msg = f"⚠️ {exc} — it may have been handled already. The queue has been refreshed."
            pending = self.sidebar.queue_choices(s)
            return (_say(history, msg), components.review_title(len(pending) - 1, s.role),
                    gr.update(choices=pending, value=""), "", self._board_update(s))
        return _do

    def _act(self, s: ChatSession, case_id: str, approve: bool, note: str, verdict: str, priority: str,
             assignee: str) -> str:
        team = self.c.care_team
        if s.role == Role.RADIOLOGIST:
            case = team.radiology_read(case_id, s.display_name, verdict, note)
            return f"🩻 Read submitted for case **{case_id}** — _{verdict}_. Next: **{wf.stage_label(case)[0]}**."
        if s.role == Role.COORDINATOR:
            case = team.triage(case_id, s.display_name, priority, assignee, note)
            return (f"🧭 Case **{case_id}** triaged {wf.PRIORITY_ICON[priority]} **{priority}** and assigned to "
                    f"**{case['payload']['triage']['clinician_name']}**.")
        case = team.review(case_id, s.display_name, approve, note)
        verb = "approved and sent to" if approve else "returned to"
        return (f"{'✅' if approve else '↩️'} Case **{case_id}** {verb} **{self.sidebar.patient_name(case['patient_key'])}**."
                + (f"\n\n_Your note will also help the Insight agent on future cases:_ “{note}”" if note else ""))

    # ------------------------------------------------------------------------ polling
    def deliver_updates(self, session: Optional[ChatSession], history: History) -> Optional[History]:
        """Validated / returned cases for a patient -> chat messages (once). None = nothing new."""
        if session is None or not session.own_patient or session.busy:
            return None
        new = self.c.care_team.undelivered(session.own_patient)
        if not new:
            return None
        history = list(history or [])
        for c in new:
            history.append({"role": "assistant", "content": _delivery_message(c)})
            self.c.care_team.mark_delivered(c["case_id"])
        return history

    def poll(self, s: Optional[ChatSession], history: History):
        """Every few seconds: patients receive reviewed answers; staff see their queue refresh."""
        if s is None:
            return (SKIP,) * 5
        if s.role == Role.PATIENT:
            updated = self.deliver_updates(s, history)
            cases_html = self.sidebar.my_cases(s)
            changed = cases_html != self._shown_cases.get(s.session_id)
            self._shown_cases[s.session_id] = cases_html
            if updated is None and not changed:
                return (SKIP,) * 5
            return (updated if updated is not None else SKIP), cases_html, SKIP, SKIP, SKIP
        if has_queue(s.role):
            return (SKIP, SKIP, *self._queue_updates(s), self._board_update(s))
        return (SKIP,) * 5


def _delivery_message(case: dict) -> str:
    who = case.get("reviewer") or "your clinician"
    asked = f"_Latest question: “{case['question']}”_"
    if case["status"] == wf.VALIDATED:
        return f"### ✅ Case {case['case_id']} confirmed by {who}\n{asked}\n\n{case['final_message']}"
    note = case.get("review_note") or "Your clinician needs a little more information."
    return (f"### ↩️ Case {case['case_id']} — {who} needs more information\n{asked}\n\n{note}\n\n"
            "You can reply here (for example describe your symptoms in more detail) or attach a report with 📎 Attach "
            "— it will be added to the same case.")
