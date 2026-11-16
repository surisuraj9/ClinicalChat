"""Steps 1–3 of the agent pipeline (all rule-based, no AI, ~milliseconds):

  📄 Data Extraction → 🔬 Clinical Analysis (Lab / Radiology / Allergy / symptoms) → 🚨 Risk Detection

Each step reports live progress as Events and stores its result on the TurnContext for the steps after it."""

from __future__ import annotations

from typing import Iterator, Sequence

from clinicalchat.agents.analysis import ClinicalAnalysisAgent
from clinicalchat.agents.extraction import DataExtractionAgent
from clinicalchat.agents.risk import RiskDetectionAgent
from clinicalchat.domain.conversation import Event
from clinicalchat.domain.risk import PATIENT_GUIDANCE
from clinicalchat.domain.roles import AGENT_LABELS, AgentName
from clinicalchat.presenters.answers import safety_banner
from clinicalchat.presenters.findings import analysis_text, extraction_text, risk_text
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.safety import SafetyChecker
from clinicalchat.services.turn import TurnContext

UNREADABLE = ("🔍 **I couldn't read enough from this report to judge it**, so I can't say whether it is normal. "
              "A clinician will check it. If you feel unwell, don't wait — see a doctor.")


class AnalysisPipeline:
    def __init__(self, extraction: DataExtractionAgent, analysis: ClinicalAnalysisAgent, risk: RiskDetectionAgent,
                 patients: PatientDirectory, safety: SafetyChecker):
        self.extraction = extraction
        self.analysis = analysis
        self.risk = risk
        self.patients = patients
        self.safety = safety

    def run(self, ctx: TurnContext, upload_doc_ids: Sequence[str] = ()) -> Iterator[Event]:
        yield from self._extract(ctx, upload_doc_ids)
        yield from self._analyse(ctx)
        yield from self._score(ctx, upload_doc_ids)

    def safety_banner(self, ctx: TurnContext) -> str:
        plan = ctx.plan
        return safety_banner(plan, self.patients.get(plan.patient_key), self.safety.alerts(plan.patient_key, plan.drugs))

    def _extract(self, ctx: TurnContext, upload_doc_ids: Sequence[str]) -> Iterator[Event]:
        label, plan = AGENT_LABELS[AgentName.EXTRACTION], ctx.plan
        if not upload_doc_ids:
            yield Event("status", f"{label} — gathering records", "Selecting the relevant reports and checking quality…",
                        "pending", "agent-extraction")
        ctx.run.ext = ext = self.extraction.run(ctx.question, ctx.session.role, plan.patient_key, plan.focus,
                                                new_symptoms=plan.symptoms, upload_doc_ids=upload_doc_ids)
        if upload_doc_ids:
            yield Event("status", f"{label} — done", f"{len(upload_doc_ids)} report(s) processed", "done", "agent-extraction")
            return
        summary = extraction_text(ext, ctx.is_patient)
        yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-extraction")
        yield Event("step", label, summary, data={"agent": "extraction"})

    def _analyse(self, ctx: TurnContext) -> Iterator[Event]:
        label, plan = AGENT_LABELS[AgentName.ANALYSIS], ctx.plan
        yield Event("status", f"{label} — analysing", "Checking values against reference ranges, imaging findings, "
                    "allergies and symptoms…", "pending", "agent-analysis")
        ctx.run.analysis = an = self.analysis.run(ctx.run.ext, plan.drugs, plan.allergens,
                                                  check_allergies=plan.check_allergies)
        patient = self.patients.get(plan.patient_key)
        summary = analysis_text(an, ctx.is_patient, patient.medications if patient else ())
        yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-analysis")
        yield Event("step", label, summary, data={"agent": "analysis"})
        banner = self.safety_banner(ctx)
        if banner:
            ctx.turn.safety = banner
            yield Event("safety", "🛡️ Safety check", banner)

    def _score(self, ctx: TurnContext, upload_doc_ids: Sequence[str]) -> Iterator[Event]:
        label, run = AGENT_LABELS[AgentName.RISK], ctx.run
        yield Event("status", f"{label} — scoring", "Applying safety rules…", "pending", "agent-risk")
        run.risk = risk = self.risk.run(self.patients.get(ctx.patient_key), run.ext, run.analysis)
        summary = risk_text(risk, ctx.is_patient)
        yield Event("status", f"{label} — {risk.risk_level}", summary.split("\n")[0], "done", "agent-risk")
        yield Event("step", label, summary, data={"agent": "risk", "level": risk.risk_level})
        level = risk.clinical_level
        if ctx.is_patient and (run.ext.new_symptoms or upload_doc_ids or level != "LOW"):
            text = UNREADABLE if level == "LOW" and run.ext.quality.low_confidence else PATIENT_GUIDANCE[level]
            yield Event("guidance", "What to do now", text, data={"level": level})
