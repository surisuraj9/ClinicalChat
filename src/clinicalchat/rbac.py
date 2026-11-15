"""Role-based access control: which agents a role may use, what data it may see and how answers are written."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional


class Role(str, Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RADIOLOGIST = "radiologist"
    COORDINATOR = "coordinator"


class AgentName(str, Enum):
    LAB = "lab"
    RADIOLOGY = "radiology"
    ALLERGY = "allergy"
    COORDINATOR = "coordinator"


AGENT_LABELS = {
    AgentName.LAB: "🧪 Lab Interpreter",
    AgentName.RADIOLOGY: "🩻 Radiology Analyzer",
    AgentName.ALLERGY: "⚠️ Allergy Safety",
    AgentName.COORDINATOR: "🗂️ Care Coordinator",
}


@dataclass(frozen=True)
class RolePolicy:
    role: Role
    label: str
    agents: FrozenSet[AgentName]
    own_record_only: bool
    audience: str


POLICIES = {
    Role.PATIENT: RolePolicy(
        role=Role.PATIENT, label="Patient",
        agents=frozenset({AgentName.LAB, AgentName.RADIOLOGY, AgentName.ALLERGY}),
        own_record_only=True,
        audience=("PATIENT. Use plain, reassuring, simple language (short sentences, no jargon — explain any medical "
                  "term you must use). Do NOT give drug doses, prescriptions or a definitive diagnosis. Explain what "
                  "the results mean in everyday terms and always suggest discussing them with their doctor. Clearly "
                  "say when something needs urgent medical attention."),
    ),
    Role.CLINICIAN: RolePolicy(
        role=Role.CLINICIAN, label="Clinician",
        agents=frozenset(AgentName),
        own_record_only=False,
        audience=("CLINICIAN. Give full technical detail: exact values with reference ranges, clinical patterns, "
                  "differential diagnosis, treatment considerations and recommended follow-up tests. Be concise and "
                  "structured."),
    ),
    Role.RADIOLOGIST: RolePolicy(
        role=Role.RADIOLOGIST, label="Radiologist",
        agents=frozenset({AgentName.RADIOLOGY, AgentName.ALLERGY}),
        own_record_only=False,
        audience=("RADIOLOGIST. Focus on imaging: technique, findings by region, impression, differential diagnosis, "
                  "critical findings, comparison/further imaging recommendations and contrast-safety considerations. "
                  "Use precise radiology terminology."),
    ),
    Role.COORDINATOR: RolePolicy(
        role=Role.COORDINATOR, label="Care Coordinator",
        agents=frozenset({AgentName.COORDINATOR, AgentName.ALLERGY}),
        own_record_only=False,
        audience=("CARE COORDINATOR. Focus on care coordination across specialties: current status, outstanding "
                  "results/tests, which specialists need to be involved, follow-up scheduling, safety alerts to hand "
                  "over and next actions. Avoid deep technical interpretation; be operational and concise."),
    ),
}


def policy(role: Role | str) -> RolePolicy:
    return POLICIES[Role(role)]


def can_use(role: Role | str, agent: AgentName) -> bool:
    return agent in policy(role).agents


def can_access_patient(role: Role | str, own_patient_key: Optional[str], patient_key: Optional[str]) -> bool:
    if patient_key is None:
        return True
    pol = policy(role)
    return not pol.own_record_only or patient_key == own_patient_key
