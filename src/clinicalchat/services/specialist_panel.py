"""Runs the course AI specialists (Lab Interpreter / Radiology Analyzer / Allergy Safety) in parallel.

Only the specialists whose record type the user's role may see AND that have data this turn are called, so a
clinician never sends imaging to the AI and a radiologist never sends labs or the allergy record. Every call is a
child "agent" observation of the 🔬 Clinical Analysis span in Langfuse. The inputs are built from the reports, not
from the question, so asking about the same reports again is answered from the LLM cache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import AbstractSet, Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from clinicalchat.agents.analysis import AnalysisResult
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.agents.specialists.ai import (AllergyReview, AllergySafetyAgent, LabInterpreterAgent, LabPanel,
                                                RadiologyAnalyzerAgent, RadiologyReview, SpecialistAgent,
                                                SpecialistReport)
from clinicalchat.domain.models import Domain, PatientProfile
from clinicalchat.infrastructure.tracing import Span
from clinicalchat.services.observability import end_agent_span

MAX_LAB_LINES = 40
MAX_STUDY_CHARS = 1800


def lab_report(an: AnalysisResult, patient: Optional[PatientProfile]) -> str:
    if not an.lab_results:
        return ""
    abnormal = {id(r) for r, _ in an.abnormal}
    ordered = sorted(an.lab_results, key=lambda rs: id(rs[0]) not in abnormal)[:MAX_LAB_LINES]
    head = _demographics(patient)
    return head + "\n".join(f"- {r.line()} [{src}]" for r, src in ordered)


def radiology_report(an: AnalysisResult) -> str:
    return "\n\n".join(f"STUDY: {f.section.title} [{f.source}]\n{f.section.text[:MAX_STUDY_CHARS]}" for f in an.imaging[:3])


def allergy_report(an: AnalysisResult, patient: Optional[PatientProfile]) -> str:
    if not (an.allergy_record_visible and patient and patient.allergies):
        return ""
    lines = [f"- {a.allergen} ({a.category}; severity {a.severity}; reactions: {', '.join(a.reactions) or 'not stated'}"
             f"{'; emergency medication: ' + a.emergency_medication if a.emergency_medication else ''})"
             for a in patient.allergies[:6]]
    kg = an.allergy_facts if an.allergy_facts and an.allergy_facts != "None." else ""
    alerts = [f"- {a.level}: {a.drug} vs {a.against} — {a.message}" for a in (an.drug_alerts + an.medication_alerts)[:6]]
    return (_demographics(patient) + "DOCUMENTED ALLERGIES:\n" + "\n".join(lines)
            + (f"\n\nKNOWLEDGE GRAPH:\n{kg}" if kg else "") + ("\n\nSAFETY ALERTS:\n" + "\n".join(alerts) if alerts else ""))


def _demographics(patient: Optional[PatientProfile]) -> str:
    if not patient or not (patient.age or patient.gender):
        return ""
    return f"Patient: age {patient.age or 'unknown'}, sex {patient.gender or 'unknown'}\n\n"


class SpecialistPanel:
    def __init__(self, llm: BaseChatModel, enabled: bool):
        self.enabled = enabled
        self.lab = LabInterpreterAgent(llm)
        self.radiology = RadiologyAnalyzerAgent(llm)
        self.allergy = AllergySafetyAgent(llm)
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="specialist")

    def jobs(self, an: AnalysisResult, patient: Optional[PatientProfile],
             domains: AbstractSet[Domain]) -> Dict[str, tuple[SpecialistAgent, str]]:
        """field on SpecialistReport -> (agent, report text) for the specialists allowed and with data."""
        candidates = {"lab": (Domain.LAB, self.lab, lab_report(an, patient)),
                      "radiology": (Domain.RADIOLOGY, self.radiology, radiology_report(an)),
                      "allergy": (Domain.ALLERGY, self.allergy, allergy_report(an, patient))}
        return {key: (agent, text) for key, (domain, agent, text) in candidates.items() if domain in domains and text}

    def run(self, an: AnalysisResult, patient: Optional[PatientProfile], domains: AbstractSet[Domain], parent: Span,
            llm_config: Callable[[Span], dict]) -> Optional[SpecialistReport]:
        """Call the permitted specialists in parallel; None when switched off or there is nothing to analyse."""
        jobs = self.jobs(an, patient, domains) if self.enabled else {}
        if not jobs:
            return None
        spans = {key: parent.child(agent.label, as_type="agent", input={"report": text},
                                   metadata={"model_purpose": "specialist"}) for key, (agent, text) in jobs.items()}
        futures = {key: self._pool.submit(agent.run, text, llm_config(spans[key])) for key, (agent, text) in jobs.items()}
        report = SpecialistReport()
        for key, future in futures.items():
            result = future.result()
            self._record(report, key, result, spans[key])
        return report

    @staticmethod
    def _record(report: SpecialistReport, key: str, result: AgentResult, span: Span) -> None:
        end_agent_span(span, result)
        report.results.append(result)
        report.timings[result.label] = result.timing()
        if result.error:
            report.errors[result.label] = result.error
            return
        model = {"lab": LabPanel, "radiology": RadiologyReview, "allergy": AllergyReview}[key]
        parsed = model.model_validate(result.output or {})
        setattr(report, key, parsed)
        confidences = _confidences(parsed)
        if confidences:
            span.score(f"{result.agent.value}_confidence", round(mean(confidences), 3), data_type="NUMERIC")


def _confidences(parsed: object) -> List[float]:
    items = next(iter(parsed.model_dump().values()), []) if hasattr(parsed, "model_dump") else []
    return [float(i["confidence"]) for i in items if isinstance(i, dict) and isinstance(i.get("confidence"), (int, float))]
