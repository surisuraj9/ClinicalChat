"""Runs the 💡 Insight Generation agent: for staff while they wait, for the care team in the background once a
patient case is opened (the patient never waits for it), with a per-patient cache keyed on the evidence.
In the background run the course AI specialists go first (when switched on), so the care team's case shows their
structured opinions and the Insight can use them."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from clinicalchat.agents.analysis import AnalysisResult
from clinicalchat.agents.extraction import ExtractionResult
from clinicalchat.agents.insight import InsightGenerationAgent
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.domain import workflow as wf
from clinicalchat.domain.risk import RiskAssessment
from clinicalchat.domain.roles import Role, policy
from clinicalchat.domain.web_evidence import WebSearchOutcome
from clinicalchat.infrastructure.repositories.caches import InsightCache
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.feedback import FeedbackRepository
from clinicalchat.infrastructure.tracing import Span, activate
from clinicalchat.presenters.findings import extraction_brief, risk_brief
from clinicalchat.services.observability import end_agent_span
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.records import RecordRetriever
from clinicalchat.services.specialist_panel import SpecialistPanel
from clinicalchat.services.turn import PipelineRun, TurnContext

WEB_EVIDENCE_HEADER = ("\n\n## WEB EVIDENCE (trusted guideline sites; cite as [W1], [W2]… for guideline / treatment "
                       "statements; it never overrides this patient's records)\n")


def review_reasons(ext: ExtractionResult, an: AnalysisResult, risk: RiskAssessment,
                   insight: Optional[AgentResult]) -> List[str]:
    """Why a clinician must look at this before anyone relies on it ("Review Required")."""
    reasons = []
    output = (insight.output if insight else None) or {}
    conf = output.get("confidence")
    if insight is not None and (insight.error or not insight.parsed):
        reasons.append("AI output could not be fully validated")
    if isinstance(conf, (int, float)) and conf < wf.REVIEW_CONFIDENCE:
        reasons.append(f"Low confidence ({conf:.0%})")
    if ext.quality.low_confidence:
        reasons.append("Low-confidence data extraction")
    if not ext.sufficient:
        reasons.append("Insufficient data")
    if an.inconsistencies:
        reasons.append("Clinical inconsistency")
    if risk.escalate:
        reasons.append(f"Risk {risk.risk_level} — escalated for clinician validation")
    return reasons


class InsightService:
    def __init__(self, agent: InsightGenerationAgent, patients: PatientDirectory, records: RecordRetriever,
                 feedback: FeedbackRepository, cases: CaseRepository, cache: InsightCache,
                 specialists: Optional[SpecialistPanel] = None):
        self.agent = agent
        self.specialists = specialists
        self.patients = patients
        self.records = records
        self.feedback = feedback
        self.cases = cases
        self.cache = cache
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="insight")
        self._background: List[Future] = []

    def write(self, question: str, audience: str, patient_key: Optional[str], run: PipelineRun, history: str,
              span: Span, config: dict, web: Optional[WebSearchOutcome] = None) -> AgentResult:
        """One Insight call; ``span`` is its Langfuse observation (record retrieval is nested under it)."""
        ext, an = run.ext, run.analysis
        with activate(span):
            context = self.records.records_context(question, ext.sections, ext.focus)
        if web and web.results:
            context += WEB_EVIDENCE_HEADER + web.context()
        return self.agent.run(question=question, audience=audience, patient=self.patients.summary(patient_key),
                              extraction=extraction_brief(ext), facts=an.facts_text(), risk=risk_brief(run.risk),
                              feedback=self.feedback.prompt_text(), context=context, history=history, config=config)

    def cached(self, patient_key: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(patient_key, fingerprint)

    def rule_based(self, question: str, run: PipelineRun) -> Dict[str, Any]:
        """What the Communication agent writes from when no AI clinical summary is needed or ready yet:
        the deterministic findings plus a short excerpt of the relevant records."""
        an, ext = run.analysis, run.ext
        excerpt = self.records.records_context(question, ext.sections[:3], ext.focus, reference_k=1) if ext else ""
        return {"key_findings": an.facts_text().splitlines()[:25] if an else [],
                "possible_conditions": an.possible_conditions[:6] if an else [],
                "records_excerpt": excerpt[:2500], "source": "rule-based findings (validated code)"}

    def in_background(self, case_id: str, ctx: TurnContext, history: str) -> None:
        """Write the clinician-facing summary for a patient case without making the patient wait."""
        self.cases.patch_payload(case_id, lambda p: p.update(insight_pending=True, insight_error=""))
        audience = policy(Role.CLINICIAN).audience
        key, run, question = ctx.patient_key, ctx.run, ctx.question
        span = ctx.trace.child("insight-generation (background, for the care team)", as_type="agent",
                               input={"question": question}, metadata={"case_id": case_id})
        config = ctx.llm_config(span)

        def work() -> None:
            self._specialists_for_case(case_id, ctx)
            ins = self.write(question, audience, key, run, history, span, config)
            end_agent_span(span, ins)
            if ins.error or not ins.parsed:
                error = ins.error or "AI output could not be validated"
                self.cases.patch_payload(case_id, lambda p: wf.attach_insight(p, None, error))
                return
            self.cache.save(key, run.evidence_fp, ins.output or {})
            self.cases.patch_payload(case_id, lambda p: wf.attach_insight(p, ins.output or {}))

        self._background.append(self._pool.submit(work))

    def _specialists_for_case(self, case_id: str, ctx: TurnContext) -> None:
        panel, an = self.specialists, ctx.run.analysis
        if not (panel and panel.enabled and an is not None and an.ai is None):
            return
        span = ctx.trace.child("clinical-analysis · AI specialists (background, for the care team)", as_type="agent",
                               metadata={"case_id": case_id})
        an.ai = report = panel.run(an, self.patients.get(ctx.patient_key), policy(ctx.session.role).domains, span,
                                   ctx.llm_config)
        span.end({"ran": report.ran if report else [], "errors": report.errors if report else {}})
        if report is not None:
            urgent = [f"{text} ({agent})" for level, text, agent in report.urgent() if level == "HIGH"]
            self.cases.patch_payload(case_id, lambda p: wf.attach_specialists(p, report.model_dump(), urgent))

    def drain(self, timeout: float = 600) -> None:
        """Wait for background work (Insight for the care team). Used by tests and on shutdown."""
        for f in list(self._background):
            f.result(timeout=timeout)
        self._background.clear()
