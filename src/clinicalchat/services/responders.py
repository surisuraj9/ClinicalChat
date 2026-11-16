"""After steps 1–3, who answers and how:

* StaffResponder   — 💡 Insight Generation while the clinician waits (+ 🌐 guideline evidence when asked for),
                     then optionally a patient-friendly draft.
* PatientResponder — ONE AI call (💬 Communication). The clinician-facing Insight is reused when nothing changed, or
                     written in the background once a case is opened, so the patient never waits for it.
* GeneralResponder — a general health question from a patient: FAQ cache → trusted web + reference material → answer.
"""

from __future__ import annotations

from typing import Iterator

from clinicalchat.agents.communication import communication_markdown
from clinicalchat.agents.guard import blocking, check_patient_text
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.domain.conversation import Event
from clinicalchat.domain.roles import AGENT_LABELS, AgentName, policy
from clinicalchat.infrastructure.repositories.caches import FaqCache
from clinicalchat.infrastructure.tracing import Tracer
from clinicalchat.presenters.answers import general_answer_footer, render_insight
from clinicalchat.services.answer_writer import AnswerWriter
from clinicalchat.services.cases import CaseFiler
from clinicalchat.services.communicator import Communicator
from clinicalchat.services.insight_service import InsightService, review_reasons
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.records import RecordRetriever
from clinicalchat.services.turn import TurnContext
from clinicalchat.services.web_enrichment import WEB_WAIT_PATIENT, WebEnrichment, guideline_query, wants_guidelines

INSIGHT = AGENT_LABELS[AgentName.INSIGHT]
COMMUNICATION = AGENT_LABELS[AgentName.COMMUNICATION]


def _remember(ctx: TurnContext, output: dict) -> None:
    """Kept on the session so "explain this to the patient" can reuse it without a new Insight call."""
    run = ctx.run
    ctx.session.last_insight = {"patient_key": ctx.patient_key, "question": ctx.question, "output": output,
                                "risk": run.risk.risk_level, "facts": run.analysis.facts_text()}


class StaffResponder:
    def __init__(self, insight: InsightService, web: WebEnrichment, communicator: Communicator, tracer: Tracer):
        self.insight = insight
        self.web = web
        self.communicator = communicator
        self.tracer = tracer

    def respond(self, ctx: TurnContext) -> Iterator[Event]:
        run, question = ctx.run, ctx.question
        yield Event("status", f"{INSIGHT} — writing", "Consolidating the findings (AI)…", "pending", "agent-insight")
        web = None
        if wants_guidelines(question):
            web = yield from self.web.search("guideline", guideline_query(question, run.analysis.possible_conditions), ctx)
        ins = self.insight.write(question, policy(ctx.session.role).audience, ctx.patient_key, run,
                                 ctx.session.history_text(), self.tracer.config_for(ctx.session), web=web)
        run.llm.append(ins)
        out = ins.output or {}
        run.review_reasons = review_reasons(run.ext, run.analysis, run.risk, ins)
        yield Event("status", f"{INSIGHT} — {'failed' if ins.error else 'done'}", ins.timing(), "done", "agent-insight")
        _remember(ctx, out)
        md = render_insight(ins, run.review_reasons, run.ext.sources)
        if web and web.results and not ins.error:
            md += web.links_markdown("🌐 Guideline / evidence sources (web — verify before acting)")
        ctx.turn.add_answer(INSIGHT, md, str(out.get("answer", ins.error)))
        yield Event("answer", INSIGHT, md, data={"agent": "insight", "review": run.review_reasons})
        if ctx.plan.mode == "explain" and not ins.error:
            yield from self.communicator.communicate(ctx, out, run.risk.clinical_level, run.abnormal)


class PatientResponder:
    def __init__(self, insight: InsightService, cases: CaseFiler, communicator: Communicator, tracer: Tracer):
        self.insight = insight
        self.cases = cases
        self.communicator = communicator
        self.tracer = tracer

    def respond(self, ctx: TurnContext, upload_doc_ids=()) -> Iterator[Event]:
        run, plan, key = ctx.run, ctx.plan, ctx.patient_key
        cached = None
        if key:
            run.evidence_fp = self.cases.evidence_fp(key, run.risk.risk_level)
            run.case_fp = self.cases.evidence_fp(key, run.risk.risk_level, reads=False)
            run.file_case = self.cases.needs_case(ctx, upload_doc_ids)
            if not (upload_doc_ids or run.new_symptoms or plan.drugs or plan.allergens):
                cached = self.insight.cached(key, run.evidence_fp)
        reused = AgentResult(agent=AgentName.INSIGHT, label=INSIGHT, output=cached) if cached is not None else None
        run.review_reasons = review_reasons(run.ext, run.analysis, run.risk, reused)
        background = bool(key and run.file_case and cached is None)
        yield self._insight_status(cached is not None, background)
        basis = cached or self.insight.rule_based(ctx.question, run)
        _remember(ctx, basis)
        yield from self.communicator.communicate(ctx, basis, run.risk.clinical_level, run.abnormal)
        if background and ctx.turn.case_id:
            self.insight.in_background(ctx.turn.case_id, ctx, ctx.session.history_text(),
                                       self.tracer.config_for(ctx.session))

    @staticmethod
    def _insight_status(reused: bool, background: bool) -> Event:
        if reused:
            return Event("status", f"{INSIGHT} — reused", "Nothing new since the last assessment — reusing it (no AI call)",
                         "done", "agent-insight")
        if background:
            return Event("status", f"{INSIGHT} — for the care team", "The clinical summary for your care team is written "
                         "in the background — you don't wait for it", "done", "agent-insight")
        return Event("status", f"{INSIGHT} — not needed", "Your question is answered from the findings above "
                     "(no clinical summary needed)", "done", "agent-insight")


class GeneralResponder:
    """General health question from a patient: informational answer, no personal records, no review needed."""

    def __init__(self, writer: AnswerWriter, web: WebEnrichment, records: RecordRetriever, patients: PatientDirectory,
                 faq: FaqCache):
        self.writer = writer
        self.web = web
        self.records = records
        self.patients = patients
        self.faq = faq

    def respond(self, ctx: TurnContext) -> Iterator[Event]:
        question = ctx.question
        allergies = self.patients.allergy_text(ctx.patient_key)
        hit = self.faq.lookup(question, allergies)
        if hit:
            yield Event("status", f"{COMMUNICATION} — instant", f"Answered before (“{hit['question'][:60]}”) — reused, "
                        "no AI call", "done", "agent-communication")
            ctx.turn.add_answer(COMMUNICATION, hit["answer"], hit["answer"])
            yield Event("answer", COMMUNICATION, hit["answer"], data={"agent": "communication", "faq_cache": True})
            return
        web = yield from self.web.search("health", question, ctx, wait=WEB_WAIT_PATIENT)
        context, sources = self.records.reference_context(question)
        if web and web.results:
            context += "\n\n## Trusted health websites (current, public information)\n" + web.context()
        yield Event("status", f"{COMMUNICATION} — answering", "Looking up reference information and writing an answer (AI)…",
                    "pending", "agent-communication")
        com = yield from self.writer.write("💬 Writing your answer…", ctx.session, False, question=question,
                                           patient=f"Documented allergies: {allergies}",
                                           risk="Not applicable (general question).", insight=context, general=True)
        ctx.run.llm.append(com)
        yield Event("status", f"{COMMUNICATION} — {'failed' if com.error else 'done'}", com.timing(), "done",
                    "agent-communication")
        if com.error:
            md = f"_The AI assistant is busy right now — please try again shortly._ ({com.error[:120]})"
        else:
            md = communication_markdown(com.output or {}) + general_answer_footer(sources)
            if web and web.results:
                md += web.links_markdown("🌐 From trusted health sites")
            if com.parsed and not blocking(check_patient_text([md], False)):
                self.faq.save(question, allergies, md)
        ctx.turn.add_answer(COMMUNICATION, md, md)
        yield Event("answer", COMMUNICATION, md, data={"agent": "communication"})
