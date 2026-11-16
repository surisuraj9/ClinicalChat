"""The care team's actions on a case — 🩻 radiology read, 🧭 triage, 👩‍⚕️ clinician review — plus their work queues and
the patient's delivery of confirmed answers. Every action is checked by the workflow rules and written to the audit log."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.personas import PERSONAS
from clinicalchat.domain.roles import Role
from clinicalchat.infrastructure.audit_log import AuditLog
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.feedback import FeedbackRepository

Case = Dict[str, Any]


class CareTeamService:
    def __init__(self, cases: CaseRepository, feedback: FeedbackRepository, audit: AuditLog):
        self.cases = cases
        self.feedback = feedback
        self.audit = audit

    # ------------------------------------------------------------------ actions
    def radiology_read(self, case_id: str, reader: str, verdict: str, report: str) -> Case:
        """🩻 The radiologist's read of the case's imaging; the case then goes to the care coordinator (or straight back
        to the assigned clinician if it was triaged already and nothing critical was found)."""
        case = self.cases.require(case_id)
        if verdict not in wf.RADIOLOGY_VERDICTS:
            raise ValueError(f"Unknown verdict {verdict}")
        doc_ids = [i["doc_id"] for i in case["payload"].get("imaging", []) if i.get("doc_id")]
        payload, status, risk = wf.radiology_read(case, reader=reader, verdict=verdict, report=report.strip(),
                                                  doc_ids=doc_ids)
        case = self.cases.update(case_id, {"payload": payload, "status": status, **({"risk_level": risk} if risk else {})})
        self.audit.record({"event": "radiology_read", "case": case_id, "reader": reader, "verdict": verdict,
                           "next": case["status"]})
        return case

    def triage(self, case_id: str, coordinator: str, priority: str, clinician: str, note: str = "") -> Case:
        """🧭 The care coordinator sets the priority and assigns the clinician who will confirm the case."""
        doctor = PERSONAS.get(clinician)
        if not doctor or doctor.role != Role.CLINICIAN:
            raise ValueError(f"{clinician} is not a clinician")
        payload = wf.triage(self.cases.require(case_id), coordinator=coordinator, priority=priority,
                            clinician=doctor.username, clinician_name=doctor.display_name, note=note.strip())
        case = self.cases.update(case_id, {"payload": payload, "status": wf.CLINICIAN})
        self.audit.record({"event": "triage", "case": case_id, "coordinator": coordinator, "priority": priority,
                           "clinician": doctor.username})
        return case

    def review(self, case_id: str, reviewer: str, approve: bool, note: str = "", edited_message: str = "") -> Case:
        """👩‍⚕️ The clinician confirms the AI assessment (VALIDATED) or asks the patient for more (RETURNED). The note is
        kept as feedback for later Insight prompts (learnability)."""
        case = self.cases.require(case_id)
        values = wf.clinician_review(case, reviewer=reviewer, approve=approve, note=note, edited_message=edited_message)
        self.cases.update(case_id, values)
        topic = ", ".join(case["payload"].get("possible_conditions", [])[:3]) or case["question"][:80]
        self.feedback.add(case_id, reviewer, values["status"], values["review_note"], topic)
        self.audit.record({"event": "review", "case": case_id, "reviewer": reviewer, "status": values["status"],
                           "note": values["review_note"], "edited": bool(edited_message.strip())})
        return self.cases.require(case_id)

    # ------------------------------------------------------------------ queues and delivery
    def case(self, case_id: str) -> Optional[Case]:
        return self.cases.get(case_id)

    def list(self, status: Optional[str] = None, patient_key: Optional[str] = None) -> List[Case]:
        return self.cases.list(status=status, patient_key=patient_key)

    def undelivered(self, patient_key: str) -> List[Case]:
        """Confirmed / returned cases the patient has not seen in the chat yet."""
        return self.cases.undelivered(patient_key)

    def mark_delivered(self, case_id: str) -> None:
        self.cases.mark_delivered(case_id)
