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
from clinicalchat.presenters.answers import general_answer_footer, render_insight
from clinicalchat.services.answer_writer import AnswerWriter
from clinicalchat.services.cases import CaseFiler
from clinicalchat.services.communicator import Communicator
from clinicalchat.infrastructure.tracing import activate
from clinicalchat.services.insight_service import InsightService, review_reasons
from clinicalchat.services.observability import end_agent_span
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
    def __init__(self, insight: InsightService, web: WebEnrichment, communicator: Communicator):
        self.insight = insight
        self.web = web
        self.communicator = communicator

    def prefetch(self, ctx: TurnContext) -> None:
        """Called by the pipeline once the rule-based findings are known: start the guideline search now, so it runs
        while the AI specialists work instead of after them."""
        if wants_guidelines(ctx.question):
            ctx.run.web_prefetch = self.web.start("guideline", self._guideline_query(ctx), ctx)

    def respond(self, ctx: TurnContext) -> Iterator[Event]:
        run, question = ctx.run, ctx.question
        yield Event("status", f"{INSIGHT} — writing", "Consolidating the findings (AI)…", "pending", "agent-insight")
        web = None
        if wants_guidelines(question):
            web = yield from self.web.search("guideline", self._guideline_query(ctx), ctx, started=run.web_prefetch)
        span = ctx.trace.child("insight-generation", as_type="agent", input={"question": question},
                               metadata={"audience": ctx.session.role.value})
        ins = self.insight.write(question, policy(ctx.session.role).audience, ctx.patient_key, run,
                                 ctx.session.history_text(), span, ctx.llm_config(span), web=web)
        end_agent_span(span, ins)
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

    @staticmethod
    def _guideline_query(ctx: TurnContext) -> str:
        return guideline_query(ctx.question, ctx.run.analysis.possible_conditions)


class PatientResponder:
    def __init__(self, insight: InsightService, cases: CaseFiler, communicator: Communicator):
        self.insight = insight
        self.cases = cases
        self.communicator = communicator

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
        with activate(ctx.trace):
            basis = cached or self.insight.rule_based(ctx.question, run)
        _remember(ctx, basis)
        yield from self.communicator.communicate(ctx, basis, run.risk.clinical_level, run.abnormal)
        if background and ctx.turn.case_id:
            self.insight.in_background(ctx.turn.case_id, ctx, ctx.session.history_text())

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
            ctx.trace.child("faq-cache", as_type="tool", input={"question": question}).end({"matched": hit["question"]})
            yield Event("status", f"{COMMUNICATION} — instant", f"Answered before (“{hit['question'][:60]}”) — reused, "
                        "no AI call", "done", "agent-communication")
            ctx.turn.add_answer(COMMUNICATION, hit["answer"], hit["answer"])
            yield Event("answer", COMMUNICATION, hit["answer"], data={"agent": "communication", "faq_cache": True})
            return
        web = yield from self.web.search("health", question, ctx, wait=WEB_WAIT_PATIENT)
        with activate(ctx.trace):
            context, sources = self.records.reference_context(question)
        if web and web.results:
            context += "\n\n## Trusted health websites (current, public information)\n" + web.context()
        yield Event("status", f"{COMMUNICATION} — answering", "Looking up reference information and writing an answer (AI)…",
                    "pending", "agent-communication")
        span = ctx.trace.child("patient-communication", as_type="agent", input={"question": question},
                               metadata={"general_question": True, "web_sources": len(web.results) if web else 0})
        com = yield from self.writer.write("💬 Writing your answer…", ctx.llm_config(span), False, question=question,
                                           patient=f"Documented allergies: {allergies}",
                                           risk="Not applicable (general question).", insight=context, general=True)
        end_agent_span(span, com)
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
