"""ChatService — answers one chat message, streaming live progress as Events:

  🧭 route (keywords, RBAC) → 🌐 unknown-medicine look-up → one handler per mode:
     status  : "where is my case?"           — from the case workflow, no AI
     general : a patient's general question  — 💬 Communication only
     explain : staff "explain to the patient" — 💬 Communication on the last insight
     case    : everything else               — 📄 → 🔬 → 🚨 (rules) → 💡 staff / 💬 patient (AI)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterator, List, Optional

from clinicalchat.domain.conversation import ChatSession, Event, Turn
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.patient_facts import PatientFactRepository
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.presenters.case_status import case_status_markdown
from clinicalchat.presenters.findings import route_text
from clinicalchat.services.communicator import Communicator
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.pipeline import AnalysisPipeline
from clinicalchat.services.responders import GeneralResponder, PatientResponder, StaffResponder
from clinicalchat.services.router import RoutePlan, Router
from clinicalchat.services.turn import TurnContext
from clinicalchat.services.turn_recorder import TurnRecorder
from clinicalchat.services.web_enrichment import WebEnrichment


def routed_event(plan: RoutePlan, patients: PatientDirectory) -> Event:
    return Event("status", "🧭 Routed", route_text(plan, patients.get(plan.patient_key)), "done", "route",
                 {"plan": plan.as_dict()})


class SymptomRecorder:
    """Symptoms a patient mentions are saved to their record (database + knowledge graph)."""

    def __init__(self, facts: PatientFactRepository, patients: PatientDirectory, kg: MedicalKnowledgeGraph):
        self.facts = facts
        self.patients = patients
        self.kg = kg

    def save(self, key: str, symptoms: List[str]) -> List[str]:
        known = set(self.kg.patient_symptoms(key))
        new = [s for s in dict.fromkeys(symptoms) if s not in known]
        for s in new:
            self.facts.add(key, "symptom", s)
            self.patients.apply_fact(key, "symptom", s)
        return new


class ChatService:
    def __init__(self, router: Router, patients: PatientDirectory, web: WebEnrichment, pipeline: AnalysisPipeline,
                 staff: StaffResponder, patient: PatientResponder, general: GeneralResponder,
                 communicator: Communicator, symptoms: SymptomRecorder, cases: CaseRepository, recorder: TurnRecorder):
        self.router = router
        self.patients = patients
        self.web = web
        self.pipeline = pipeline
        self.staff = staff
        self.patient = patient
        self.general = general
        self.communicator = communicator
        self.symptoms = symptoms
        self.cases = cases
        self.recorder = recorder
        self._handlers: Dict[str, Callable[[TurnContext], Iterator[Event]]] = {
            "status": self._status, "general": self._general, "explain": self._explain, "case": self._case}

    def ask(self, question: str, session: ChatSession) -> Iterator[Event]:
        question = question.strip()
        yield Event("status", "🧭 Understanding your question", "Detecting the topic and checking access…", "pending", "route")
        context_patient = session.selected_patient or session.last_patient  # staff follow-ups keep the patient
        current = session.own_patient or context_patient
        has_insight = bool(session.last_insight and session.last_insight.get("patient_key") == current)
        plan = self.router.route(question, session.role, session_patient=context_patient,
                                 own_patient=session.own_patient, has_last_insight=has_insight)
        ctx = TurnContext(question, session, plan, Turn(question=question, role=session.role.value,
                                                        patient_key=plan.patient_key))
        yield routed_event(plan, self.patients)
        if plan.blocked:
            ctx.turn.blocked = plan.blocked
            yield Event("blocked", "🔒 Access", plan.blocked)
        else:
            if plan.mode != "status":
                yield from self.web.lookup_unknown_drugs(ctx)
            mode = "case" if plan.mode == "explain" and not has_insight else plan.mode
            yield from self._handlers[mode](ctx)
        self.recorder.finish(ctx)

    def run(self, question: str, session: ChatSession) -> List[Event]:
        return list(self.ask(question, session))

    def case_status_markdown(self, patient_key: Optional[str]) -> str:
        return case_status_markdown(self.cases.list(patient_key=patient_key) if patient_key else [])

    # ------------------------------------------------------------------ mode handlers
    def _status(self, ctx: TurnContext) -> Iterator[Event]:
        md = self.case_status_markdown(ctx.patient_key)
        ctx.turn.add_answer("📁 Case status", md, md)
        yield Event("answer", "📁 Your case — where it is now", md, data={"agent": "status"})

    def _general(self, ctx: TurnContext) -> Iterator[Event]:
        banner = self.pipeline.safety_banner(ctx)
        if banner:
            ctx.turn.safety = banner
            yield Event("safety", "🛡️ Safety check", banner)
        yield from self.general.respond(ctx)

    def _explain(self, ctx: TurnContext) -> Iterator[Event]:
        """Staff: re-use the last clinical insight for a patient-friendly explanation (no new Insight call)."""
        last = ctx.session.last_insight or {}
        about = replace(ctx, plan=RoutePlan(mode="explain", patient_key=last.get("patient_key")))
        abnormal = "outside reference range" in last.get("facts", "") or last.get("risk") in ("HIGH", "CRITICAL")
        yield from self.communicator.communicate(about, last.get("output", {}), last.get("risk", "LOW"), abnormal,
                                                 question=last.get("question", ctx.question))

    def _case(self, ctx: TurnContext) -> Iterator[Event]:
        if ctx.is_patient and ctx.plan.symptoms and ctx.patient_key:
            ctx.run.new_symptoms = self.symptoms.save(ctx.patient_key, ctx.plan.symptoms)
        yield from self.pipeline.run(ctx)
        yield from (self.patient.respond(ctx) if ctx.is_patient else self.staff.respond(ctx))
