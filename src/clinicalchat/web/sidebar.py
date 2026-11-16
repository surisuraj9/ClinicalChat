"""Read-model for the sidebar: what each role sees about patients and cases (queries the container, returns HTML)."""

from __future__ import annotations

from typing import Dict, List, Optional

from clinicalchat.container import Container
from clinicalchat.domain import workflow as wf
from clinicalchat.domain.conversation import ChatSession
from clinicalchat.domain.roles import Role, policy
from clinicalchat.web import components

PRIORITY_RANK = {p: i for i, p in enumerate(wf.PRIORITIES)}


def has_queue(role: Optional[Role]) -> bool:
    return role in wf.QUEUE_FOR_ROLE


class Sidebar:
    def __init__(self, container: Container):
        self._patients = container.kb.patients
        self._kg = container.kb.kg
        self._care_team = container.care_team

    def patient_name(self, key: str) -> str:
        p = self._patients.get(key)
        return p.name if p else key

    def patient_card(self, session: Optional[ChatSession]) -> str:
        if session is None:
            return ""
        p = self._patients.get(session.own_patient or session.selected_patient)
        if not p:
            return components.patient_card(None, [], [], False)
        return components.patient_card(p, self._patients.domains_of(p.patient_key),
                                       self._kg.patient_symptoms(p.patient_key), policy(session.role).can_upload)

    def my_cases(self, session: Optional[ChatSession]) -> str:
        if session is None or not session.own_patient:
            return ""
        return components.my_cases_html(self._care_team.list(patient_key=session.own_patient))

    def staff_patient_choices(self) -> List[tuple]:
        return [("— No specific patient —", "")] + [
            (f"{p.label} · {', '.join(d.value.replace('_', ' ') for d in self._patients.domains_of(p.patient_key)) or 'new patient'}",
             p.patient_key) for p in self._patients.all()]

    def queue(self, session: Optional[ChatSession]) -> List[dict]:
        """The role's work queue; clinicians only see cases assigned to them (or legacy un-triaged ones)."""
        if session is None or not has_queue(session.role):
            return []
        cases = self._care_team.list(status=wf.QUEUE_FOR_ROLE[session.role])
        if session.role == Role.CLINICIAN:
            cases = [c for c in cases if wf.assigned_clinician(c) in (None, session.username)]
            cases.sort(key=lambda c: PRIORITY_RANK.get((c["payload"].get("triage") or {}).get("priority"), 1))
        return cases

    def queue_choices(self, session: Optional[ChatSession]) -> List[tuple]:
        cases = self.queue(session)
        names: Dict[str, str] = {c["patient_key"]: self.patient_name(c["patient_key"]) for c in cases}
        return components.queue_choices(cases, names, session.role if session else None)

    def board(self) -> str:
        return components.board_html(self._care_team.list())
