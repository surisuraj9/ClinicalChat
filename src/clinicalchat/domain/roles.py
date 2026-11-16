"""Role-based access control: which agents run for a role, which record domains it may see and how answers are written.

Record domains follow the course design (Week 5 ``role_permissions``) with one deliberate extension — a patient may
read their OWN complete record:

    patient      own record only, all domains        clinician    lab + allergy (+ patient record, reference)
    radiologist  radiology (+ reference)             coordinator  everything (care coordination)

Care-team hand-over notes on a case (the radiologist's read, the triage note, the AI case summary) are shared with
the assigned team, as in any hospital workflow; raw records outside a role's domains are not."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional

from clinicalchat.domain.models import Domain


class Role(str, Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RADIOLOGIST = "radiologist"
    COORDINATOR = "coordinator"


class AgentName(str, Enum):
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    RISK = "risk"
    INSIGHT = "insight"
    COMMUNICATION = "communication"
    # the course's AI specialists, run inside Clinical Analysis (agents/specialists/ai.py)
    LAB_INTERPRETER = "lab_interpreter"
    RADIOLOGY_ANALYZER = "radiology_analyzer"
    ALLERGY_SAFETY = "allergy_safety"


AGENT_LABELS = {
    AgentName.EXTRACTION: "📄 Data Extraction",
    AgentName.ANALYSIS: "🔬 Clinical Analysis",
    AgentName.RISK: "🚨 Risk Detection",
    AgentName.INSIGHT: "💡 Insight Generation",
    AgentName.COMMUNICATION: "💬 Patient Communication",
    AgentName.LAB_INTERPRETER: "🧪 Lab Interpreter (AI)",
    AgentName.RADIOLOGY_ANALYZER: "🩻 Radiology Analyzer (AI)",
    AgentName.ALLERGY_SAFETY: "🛡️ Allergy Safety (AI)",
}
# the five pipeline agents shown in the progress panel (the AI specialists are part of Clinical Analysis)
AGENT_ORDER = [AgentName.EXTRACTION, AgentName.ANALYSIS, AgentName.RISK, AgentName.INSIGHT, AgentName.COMMUNICATION]
ALL_DOMAINS = frozenset(Domain)
CLINICIAN_DOMAINS = frozenset({Domain.LAB, Domain.ALLERGY, Domain.PATIENT_RECORD, Domain.REFERENCE})
RADIOLOGIST_DOMAINS = frozenset({Domain.RADIOLOGY, Domain.REFERENCE})


@dataclass(frozen=True)
class RolePolicy:
    role: Role
    label: str
    agents: FrozenSet[AgentName]
    domains: FrozenSet[Domain]
    own_record_only: bool
    can_upload: bool
    audience: str


_STAFF_AGENTS = frozenset({AgentName.EXTRACTION, AgentName.ANALYSIS, AgentName.RISK, AgentName.INSIGHT})

POLICIES = {
    Role.PATIENT: RolePolicy(
        role=Role.PATIENT, label="Patient", agents=frozenset(AgentName), domains=ALL_DOMAINS,
        own_record_only=True, can_upload=True,
        audience=("PATIENT. Use plain, reassuring, simple language (short sentences, no jargon — explain any medical "
                  "term you must use). Do NOT give drug doses, prescriptions or a definitive diagnosis. Explain what "
                  "things mean in everyday terms and always suggest discussing them with their doctor. Clearly say "
                  "when something needs urgent medical attention."),
    ),
    Role.CLINICIAN: RolePolicy(
        role=Role.CLINICIAN, label="Clinician", agents=frozenset(AgentName), domains=CLINICIAN_DOMAINS,
        own_record_only=False, can_upload=True,
        audience=("CLINICIAN. Give full technical detail: exact values with reference ranges, clinical patterns, "
                  "differential diagnosis, treatment considerations and recommended follow-up tests. Be concise and "
                  "structured."),
    ),
    Role.RADIOLOGIST: RolePolicy(
        role=Role.RADIOLOGIST, label="Radiologist", agents=_STAFF_AGENTS,
        domains=RADIOLOGIST_DOMAINS,
        own_record_only=False, can_upload=False,
        audience=("RADIOLOGIST. Focus on imaging: technique, findings by region, impression, differential diagnosis, "
                  "critical findings, comparison/further imaging recommendations and contrast-safety considerations. "
                  "Use precise radiology terminology."),
    ),
    Role.COORDINATOR: RolePolicy(
        role=Role.COORDINATOR, label="Care Coordinator", agents=_STAFF_AGENTS, domains=ALL_DOMAINS,
        own_record_only=False, can_upload=False,
        audience=("CARE COORDINATOR. Focus on care coordination across specialties: current status, outstanding "
                  "results/tests, which specialists need to be involved, follow-up scheduling, safety alerts to hand "
                  "over and next actions. Avoid deep technical interpretation; be operational and concise."),
    ),
}


def policy(role: Role | str) -> RolePolicy:
    return POLICIES[Role(role)]


def can_access_patient(role: Role | str, own_patient_key: Optional[str], patient_key: Optional[str]) -> bool:
    if patient_key is None:
        return True
    return not policy(role).own_record_only or patient_key == own_patient_key
