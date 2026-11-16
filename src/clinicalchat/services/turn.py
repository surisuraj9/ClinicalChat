"""What one chat turn carries through the services: the question, who asked, the route plan and what each step found."""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional

from clinicalchat.agents.analysis import AnalysisResult
from clinicalchat.agents.extraction import ExtractionResult
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.domain.conversation import ChatSession, Turn
from clinicalchat.domain.risk import RiskAssessment
from clinicalchat.domain.roles import Role
from clinicalchat.domain.web_evidence import WebSearchOutcome
from clinicalchat.infrastructure.tracing import NULL_SPAN, Span
from clinicalchat.services.router import RoutePlan


@dataclass
class PipelineRun:
    """Everything one pipeline run produced (for the audit log and case creation)."""
    ext: Optional[ExtractionResult] = None
    analysis: Optional[AnalysisResult] = None
    risk: Optional[RiskAssessment] = None
    llm: List[AgentResult] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    new_symptoms: List[str] = field(default_factory=list)
    file_case: bool = False  # patient: open/update a care-team case (new evidence or an explicit review request)
    evidence_fp: str = ""  # Insight cache key
    case_fp: str = ""  # what a case covered (for the escalation safety net)
    web: List[WebSearchOutcome] = field(default_factory=list)
    web_prefetch: Optional[Future] = None  # staff guideline search started while the AI specialists work

    @property
    def abnormal(self) -> bool:
        return bool((self.analysis and self.analysis.abnormal) or (self.risk and self.risk.escalate))


@dataclass
class TurnContext:
    question: str
    session: ChatSession
    plan: RoutePlan
    turn: Turn
    run: PipelineRun = field(default_factory=PipelineRun)
    started: float = field(default_factory=time.perf_counter)
    trace: Span = NULL_SPAN  # the Langfuse trace of this turn (no-op when tracing is off)

    @property
    def is_patient(self) -> bool:
        return self.session.role == Role.PATIENT

    @property
    def patient_key(self) -> Optional[str]:
        return self.plan.patient_key

    def llm_config(self, span: Span) -> dict:
        """LangChain config so an agent's LLM call is recorded under ``span`` with the user's role and session."""
        return span.langchain_config(tags=[self.session.role.value],
                                     metadata={"user": self.session.username, "session_id": self.session.session_id,
                                               "patient": self.patient_key or ""})
