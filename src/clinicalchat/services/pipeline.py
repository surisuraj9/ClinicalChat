"""Steps 1–3 of the agent pipeline:

  📄 Data Extraction → 🔬 Clinical Analysis (Lab / Radiology / Allergy / symptoms) → 🚨 Risk Detection

The rule-based work takes milliseconds. For staff, Clinical Analysis also asks the course AI specialists
(services/specialist_panel.py) when they are switched on; for patients they run later, in the background, so the
patient never waits for them. Each step reports live progress as Events, stores its result on the TurnContext for the
steps after it, and is one "agent" observation in the turn's Langfuse trace."""

from __future__ import annotations

from typing import AbstractSet, Callable, Iterator, Optional, Sequence

from clinicalchat.agents.analysis import AnalysisResult, ClinicalAnalysisAgent
from clinicalchat.agents.extraction import DataExtractionAgent
from clinicalchat.agents.risk import RiskDetectionAgent
from clinicalchat.domain.conversation import Event
from clinicalchat.domain.risk import PATIENT_GUIDANCE
from clinicalchat.domain.models import Domain, PatientProfile
from clinicalchat.domain.roles import AGENT_LABELS, AgentName, policy
from clinicalchat.infrastructure.tracing import Span, activate
from clinicalchat.presenters.answers import safety_banner
from clinicalchat.presenters.findings import analysis_text, extraction_text, risk_text, specialists_text
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.safety import SafetyChecker
from clinicalchat.services.specialist_panel import SpecialistPanel
from clinicalchat.services.turn import TurnContext

UNREADABLE = ("🔍 **I couldn't read enough from this report to judge it**, so I can't say whether it is normal. "
              "A clinician will check it. If you feel unwell, don't wait — see a doctor.")


class AnalysisPipeline:
    def __init__(self, extraction: DataExtractionAgent, analysis: ClinicalAnalysisAgent, risk: RiskDetectionAgent,
                 patients: PatientDirectory, safety: SafetyChecker, specialists: Optional[SpecialistPanel] = None):
        self.extraction = extraction
        self.analysis = analysis
        self.risk = risk
        self.patients = patients
        self.safety = safety
        self.specialists = specialists

    def run(self, ctx: TurnContext, upload_doc_ids: Sequence[str] = (),
            on_analysed: Optional[Callable[[TurnContext], None]] = None) -> Iterator[Event]:
        """``on_analysed`` runs once the rule-based findings are ready (before the slower AI specialists), so the
        caller can start independent work — e.g. a guideline web search — in parallel."""
        yield from self._extract(ctx, upload_doc_ids)
        yield from self._analyse(ctx, on_analysed)
        yield from self._score(ctx, upload_doc_ids)

    def safety_banner(self, ctx: TurnContext) -> str:
        plan = ctx.plan
        return safety_banner(plan, self.patients.get(plan.patient_key), self.safety.alerts(plan.patient_key, plan.drugs),
                             show_allergy_record=Domain.ALLERGY in policy(ctx.session.role).domains)

    def _extract(self, ctx: TurnContext, upload_doc_ids: Sequence[str]) -> Iterator[Event]:
        label, plan = AGENT_LABELS[AgentName.EXTRACTION], ctx.plan
        if not upload_doc_ids:
            yield Event("status", f"{label} — gathering records", "Selecting the relevant reports and checking quality…",
                        "pending", "agent-extraction")
        span = ctx.trace.child("data-extraction", as_type="agent", input={"focus": [d.value for d in plan.focus],
                                                                          "uploads": list(upload_doc_ids)})
        with activate(span):  # record retrieval (vector search) nests under this step
            ctx.run.ext = ext = self.extraction.run(ctx.question, ctx.session.role, plan.patient_key, plan.focus,
                                                    new_symptoms=plan.symptoms, upload_doc_ids=upload_doc_ids)
        span.end({"sources": ext.sources, "symptoms": ext.symptoms, "missing": ext.missing,
                  "denied": [d.value for d in ext.denied], "quality": ext.quality.model_dump()})
        if upload_doc_ids:
            yield Event("status", f"{label} — done", f"{len(upload_doc_ids)} report(s) processed", "done", "agent-extraction")
            return
        summary = extraction_text(ext, ctx.is_patient)
        yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-extraction")
        yield Event("step", label, summary, data={"agent": "extraction"})

    def _analyse(self, ctx: TurnContext, on_analysed: Optional[Callable[[TurnContext], None]]) -> Iterator[Event]:
        label, plan = AGENT_LABELS[AgentName.ANALYSIS], ctx.plan
        yield Event("status", f"{label} — analysing", "Checking values against reference ranges, imaging findings, "
                    "allergies and symptoms…", "pending", "agent-analysis")
        span = ctx.trace.child("clinical-analysis", as_type="agent",
                               input={"drugs": plan.drugs, "allergens": plan.allergens, "symptoms": ctx.run.ext.symptoms})
        domains = policy(ctx.session.role).domains
        ctx.run.analysis = an = self.analysis.run(ctx.run.ext, plan.drugs, plan.allergens,
                                                  check_allergies=plan.check_allergies,
                                                  allergy_record=Domain.ALLERGY in domains)
        _knowledge_graph_span(span, an)
        if on_analysed:
            on_analysed(ctx)
        patient = self.patients.get(plan.patient_key)
        summary = analysis_text(an, ctx.is_patient, patient.medications if patient else ())
        yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-analysis")
        yield Event("step", label, summary, data={"agent": "analysis"})
        if not ctx.is_patient:
            yield from self._ask_specialists(ctx, span, patient, domains)
        span.end({"specialists": an.specialists, "abnormal_labs": len(an.abnormal), "imaging": len(an.imaging),
                  "possible_conditions": an.possible_conditions, "inconsistencies": an.inconsistencies,
                  "ai_specialists": an.ai.ran if an.ai else []})
        banner = self.safety_banner(ctx)
        if banner:
            ctx.turn.safety = banner
            yield Event("safety", "🛡️ Safety check", banner)

    def _ask_specialists(self, ctx: TurnContext, span: Span, patient: Optional[PatientProfile],
                         domains: AbstractSet[Domain]) -> Iterator[Event]:
        """Staff only: the course AI specialists, in parallel, for the record types this role may see."""
        panel, an = self.specialists, ctx.run.analysis
        if not (panel and panel.enabled and panel.jobs(an, patient, domains)):
            return
        label = AGENT_LABELS[AgentName.ANALYSIS]
        yield Event("status", f"{label} — AI specialists", "Lab Interpreter / Radiology Analyzer / Allergy Safety "
                    "(AI) are reviewing the reports…", "pending", "agent-analysis")
        an.ai = report = panel.run(an, patient, domains, span, ctx.llm_config)
        if report is None:
            return
        ctx.run.llm.extend(report.results)
        yield Event("status", f"{label} — done", "AI specialists: " + ", ".join(
            f"{k} {v}" for k, v in report.timings.items()), "done", "agent-analysis")
        yield Event("step", f"{label} · AI specialists", specialists_text(report), data={"agent": "specialists"})

    def _score(self, ctx: TurnContext, upload_doc_ids: Sequence[str]) -> Iterator[Event]:
        label, run = AGENT_LABELS[AgentName.RISK], ctx.run
        yield Event("status", f"{label} — scoring", "Applying safety rules…", "pending", "agent-risk")
        with ctx.trace.step("risk-detection", as_type="agent") as span:
            run.risk = risk = self.risk.run(self.patients.get(ctx.patient_key), run.ext, run.analysis)
            span.update(output={"risk_level": risk.risk_level, "escalate": risk.escalate,
                                "indicators": [f"[{i.level}] {i.indicator}" for i in risk.indicators[:12]]})
        summary = risk_text(risk, ctx.is_patient)
        yield Event("status", f"{label} — {risk.risk_level}", summary.split("\n")[0], "done", "agent-risk")
        yield Event("step", label, summary, data={"agent": "risk", "level": risk.risk_level})
        level = risk.clinical_level
        if ctx.is_patient and (run.ext.new_symptoms or upload_doc_ids or level != "LOW"):
            text = UNREADABLE if level == "LOW" and run.ext.quality.low_confidence else PATIENT_GUIDANCE[level]
            yield Event("guidance", "What to do now", text, data={"level": level})


def _knowledge_graph_span(parent: Span, an: AnalysisResult) -> None:
    """The knowledge-graph safety + symptom reasoning, recorded as a tool call under Clinical Analysis."""
    alerts = [f"{a.level}: {a.drug} vs {a.against}" for a in (an.drug_alerts + an.medication_alerts)]
    tool = parent.child("knowledge-graph", as_type="tool", input={"allergies": an.allergies})
    tool.end({"safety_alerts": alerts, "differential": [d["condition"] for d in an.differential[:5]],
              "documented_conditions": an.documented_conditions})
