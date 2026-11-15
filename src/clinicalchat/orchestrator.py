"""Orchestrator: route (no LLM) -> deterministic safety -> specialist agents in parallel (1 LLM call each) -> combine.

`ask()` is a generator of `Event`s so the UI can show live progress (routing, gathering records, analysing...).
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from clinicalchat.agents import AgentResult, SpecialistAgent
from clinicalchat.audit import Auditor
from clinicalchat.engine import ClinicalEngine, alert_line
from clinicalchat.llm import get_chat_model
from clinicalchat.models import Domain
from clinicalchat.rbac import AGENT_LABELS, AgentName, Role, policy
from clinicalchat.router import RoutePlan, Router

HISTORY_TURNS = 3
NEEDS_DOMAIN = {AgentName.LAB: Domain.LAB, AgentName.RADIOLOGY: Domain.RADIOLOGY}


@dataclass
class Turn:
    question: str
    role: str
    patient_key: Optional[str]
    agents: List[str] = field(default_factory=list)
    safety: str = ""
    answers: List[Dict[str, Any]] = field(default_factory=list)  # {label, markdown}
    blocked: str = ""
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))


@dataclass
class ChatSession:
    username: str
    role: Role
    display_name: str = ""
    own_patient: Optional[str] = None
    selected_patient: Optional[str] = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: List[Turn] = field(default_factory=list)
    last_agents: List[AgentName] = field(default_factory=list)
    last_patient: Optional[str] = None

    def history_text(self) -> str:
        lines = []
        for t in self.turns[-HISTORY_TURNS:]:
            answer = " ".join(a["summary"] for a in t.answers)[:400] if t.answers else t.blocked
            lines.append(f"Q: {t.question}\nA: {answer}")
        return "\n".join(lines)


@dataclass
class Event:
    kind: str  # status | safety | answer | blocked | done
    title: str = ""
    text: str = ""
    status: str = "done"  # pending | done
    key: str = ""  # stable id so the UI can update a status line in place
    data: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    def __init__(self, engine: ClinicalEngine, llm: Optional[BaseChatModel] = None, auditor: Optional[Auditor] = None,
                 max_workers: int = 4):
        self.engine = engine
        self.router = Router(engine)
        self.llm = llm or get_chat_model()
        self.auditor = auditor or Auditor(engine.settings)
        self.agents = {name: SpecialistAgent(name, self.llm) for name in AgentName}
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent")

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str, session: ChatSession) -> Iterator[Event]:
        started = time.perf_counter()
        question = question.strip()
        turn = Turn(question=question, role=session.role.value, patient_key=None)

        yield Event("status", "🧭 Understanding your question", "Detecting intent and checking access…", "pending", "route")
        same_patient = (session.own_patient or session.selected_patient) == session.last_patient
        plan = self.router.route(question, session.role, session_patient=session.selected_patient,
                                 own_patient=session.own_patient,
                                 previous_agents=session.last_agents if same_patient else ())
        turn.patient_key = plan.patient_key
        patient = self.engine.patient(plan.patient_key)
        who = patient.label if patient else "no specific patient"
        route_text = f"**Patient:** {who}" + (" (named in question)" if plan.patient_source == "question" else "")
        route_text += f"\n**Specialists:** {plan.describe()}" if not plan.blocked else ""
        yield Event("status", "🧭 Routed", route_text, "done", "route", {"plan": plan.__dict__})

        if plan.blocked:
            turn.blocked = plan.blocked
            yield Event("blocked", "🔒 Access", plan.blocked)
            return self._finish(session, turn, plan, [], started)

        # ---- deterministic safety first (always shown, no LLM)
        safety_md = self.safety_banner(plan)
        if safety_md:
            turn.safety = safety_md
            yield Event("safety", "🛡️ Safety check", safety_md)

        # ---- gather records per agent (fast, local)
        yield Event("status", "📚 Gathering records", "Searching patient records and the reference library…", "pending", "gather")
        jobs = {}
        audience = policy(session.role).audience
        history = session.history_text()
        no_data: List[AgentResult] = []
        domains = set(self.engine.patient_domains(plan.patient_key)) if plan.patient_key else None
        for name in plan.agents:
            needed = NEEDS_DOMAIN.get(name)
            if domains is not None and needed and needed not in domains:
                # nothing to interpret for this patient -> answer without spending an LLM call
                who_ = "your record" if session.role == Role.PATIENT else f"{patient.name}'s record"
                no_data.append(AgentResult(agent=name, label=AGENT_LABELS[name], output={
                    "answer": f"There are no {needed.value} reports on file in {who_}, so there is nothing to interpret yet. "
                              "Once a report is added I can explain it.", "sources": []}))
                continue
            context, sources, facts = self._context_for(name, question, plan)
            jobs[name] = dict(question=question, audience=audience, patient=self.engine.patient_summary(plan.patient_key),
                              facts=facts, context=context, history=history, sources=sources)
        n_sources = len({s for j in jobs.values() for s in j["sources"]})
        yield Event("status", "📚 Records gathered",
                    f"{n_sources} source document(s) selected for {len(jobs)} specialist(s)" if jobs else "No matching reports on file",
                    "done", "gather")

        # ---- run agents in parallel; stream each result as it finishes
        config = {"callbacks": self.auditor.callbacks(), "tags": [session.role.value],
                  "metadata": {"user": session.username, "session_id": session.session_id}}
        futures = {}
        for name, job in jobs.items():
            yield Event("status", f"{AGENT_LABELS[name]} — analysing", "Reading the records and preparing an answer…",
                        "pending", f"agent-{name.value}")
            futures[self.pool.submit(self.agents[name].run, config=config, **job)] = name
        results: List[AgentResult] = []
        for res in no_data:
            results.append(res)
            md = render_result(res, session.role)
            turn.answers.append({"label": res.label, "markdown": md, "summary": res.output["answer"][:300]})
            yield Event("answer", res.label, md, data={"result": res.model_dump()})
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            secs = res.latency_ms / 1000
            note = "cached answer" if res.cache_hit else f"{secs:.1f}s"
            yield Event("status", f"{res.label} — {'done' if not res.error else 'failed'}", note, "done", f"agent-{res.agent.value}")
            md = render_result(res, session.role)
            turn.answers.append({"label": res.label, "markdown": md, "summary": (res.output or {}).get("answer", res.error)[:300]})
            yield Event("answer", res.label, md, data={"result": res.model_dump()})

        return self._finish(session, turn, plan, results, started)

    def run(self, question: str, session: ChatSession) -> List[Event]:
        return list(self.ask(question, session))

    # ------------------------------------------------------------------ helpers
    def _finish(self, session: ChatSession, turn: Turn, plan: RoutePlan, results: List[AgentResult], started: float) -> None:
        turn.agents = [a.value for a in plan.agents]
        session.turns.append(turn)
        if plan.agents:
            session.last_agents = list(plan.agents)
            session.last_patient = plan.patient_key
        self.auditor.record({
            "session": session.session_id, "user": session.username, "role": session.role.value,
            "patient": plan.patient_key, "question": turn.question, "agents": turn.agents,
            "denied": [a.value for a in plan.denied], "blocked": plan.blocked,
            "drugs": plan.drugs, "allergens": plan.allergens, "safety_shown": bool(turn.safety),
            "llm_calls": sum(1 for r in results if not r.cache_hit and not r.error),
            "cache_hits": sum(1 for r in results if r.cache_hit),
            "errors": [r.error for r in results if r.error],
            "latency_ms": int((time.perf_counter() - started) * 1000),
        })

    def _context_for(self, name: AgentName, question: str, plan: RoutePlan):
        e, key = self.engine, plan.patient_key
        allergy_line = self._allergy_line(key)
        if name == AgentName.LAB:
            context, sources = e.lab_context(question, key)
            facts = "Lab flags in RECORDS (-> NORMAL/HIGH/LOW/CRITICAL) were computed from the printed reference ranges."
        elif name == AgentName.RADIOLOGY:
            context, sources = e.radiology_context(question, key)
            facts = allergy_line + ("\nIf iodinated/gadolinium contrast is relevant, comment on contrast safety." if key else "")
        elif name == AgentName.ALLERGY:
            context, sources = e.allergy_context(question, key, plan.drugs, plan.allergens)
            facts = e.allergy_facts(key, plan.drugs, plan.allergens)
        else:
            context, sources = e.coordinator_context(question, key)
            facts = allergy_line
        return context, sources, facts

    def _allergy_line(self, key: Optional[str]) -> str:
        p = self.engine.patient(key)
        if not p:
            return "No patient selected."
        if not p.allergies:
            return "No documented allergies."
        return "Documented allergies: " + ", ".join(f"{a.allergen} ({a.severity})" for a in p.allergies)

    def safety_banner(self, plan: RoutePlan) -> str:
        p = self.engine.patient(plan.patient_key)
        lines = []
        alerts = self.engine.safety_alerts(plan.patient_key, plan.drugs)
        for a in alerts:
            icon = "⛔" if a.level == "CONTRAINDICATED" else "⚠️"
            lines.append(f"{icon} **{a.level}** — {alert_line(a).split('] ', 1)[1]}")
        if plan.drugs and not alerts:
            lines.append(f"✅ No allergy or interaction conflict found for **{', '.join(plan.drugs)}**"
                         + ("" if p else " (select a patient to check their allergies)"))
        if p and p.allergies:
            lines.append("🩺 **Documented allergies:** " + "; ".join(
                f"{a.allergen} ({a.severity.lower()}{', ' + ', '.join(a.reactions[:2]) if a.reactions else ''})" for a in p.allergies))
        return "\n".join(lines)


# ---------------------------------------------------------------------- rendering
def _bullets(title: str, items) -> str:
    items = [i for i in (items or []) if str(i).strip()]
    return f"\n\n**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""


URGENCY_BADGE = {"SOON": "🟡 Soon", "URGENT": "🟠 Urgent", "EMERGENCY": "🔴 Emergency"}


def render_result(res: AgentResult, role: Role) -> str:
    if res.error:
        return f"_This specialist could not answer right now._ {res.error}"
    o = res.output or {}
    md = str(o.get("answer", "")).strip()
    urgency = URGENCY_BADGE.get(str(o.get("urgency", "")).upper())
    if role == Role.PATIENT:
        md += _bullets("What you can do next", o.get("recommendations"))
        if o.get("emergency_protocol"):
            md += f"\n\n**If you have a severe reaction:** {o['emergency_protocol']}"
    else:
        if res.agent == AgentName.LAB:
            md += _bullets("Abnormal values", o.get("abnormal_values")) + _bullets("Possible conditions", o.get("possible_conditions"))
        elif res.agent == AgentName.RADIOLOGY:
            if o.get("impression"):
                md += f"\n\n**Impression ({o.get('study') or 'study'}):** {o['impression']}"
            md += _bullets("Differential diagnosis", o.get("differential_diagnosis")) + _bullets("🚨 Critical findings", o.get("critical_findings"))
        elif res.agent == AgentName.ALLERGY:
            if o.get("risk_level"):
                md += f"\n\n**Risk level:** {o['risk_level']}"
            md += (_bullets("Cross-reactivities", o.get("cross_reactivities")) + _bullets("Avoid", o.get("avoid"))
                   + _bullets("Safe alternatives", o.get("safe_alternatives")))
            if o.get("emergency_protocol"):
                md += f"\n\n**Emergency protocol:** {o['emergency_protocol']}"
        elif res.agent == AgentName.COORDINATOR:
            if o.get("care_status") and o.get("care_status") not in md:
                md += f"\n\n**Care status:** {o['care_status']}"
            md += (_bullets("Pending items", o.get("pending_items")) + _bullets("Specialists to involve", o.get("specialists"))
                   + _bullets("Handoff notes", o.get("handoff_notes")))
        md += _bullets("Recommended follow-up", o.get("recommendations"))
    if urgency:
        md += f"\n\n**Urgency:** {urgency}"
    sources = o.get("sources") or res.sources
    if sources:
        md += f"\n\n<sub>Sources: {', '.join(dict.fromkeys(map(str, sources)))}</sub>"
    return md
