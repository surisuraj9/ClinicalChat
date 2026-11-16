"""Keyword router (no LLM): decides the pipeline mode, the record domains a question is about (with RBAC), the patient it
concerns and the symptoms / drugs / allergens it mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clinicalchat.domain.models import Domain
from clinicalchat.domain.roles import AGENT_LABELS, AGENT_ORDER, AgentName, Role, RolePolicy, can_access_patient, policy
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.patients import PatientDirectory

DOMAIN_PATTERNS = {
    Domain.LAB: (r"\blabs?\b|blood (test|work|report)|\bcbc\b|blood count|h(a)?emoglobin|\bwbc\b|platelet|"
                 r"metabolic|\bcmp\b|\bbmp\b|electrolyte|liver function|\blft\b|kidney function|renal|\bkft\b|\brft\b|"
                 r"creatinine|thyroid|\btsh\b|\bt3\b|\bt4\b|sugar|glucose|hba1c|diabet|lipid|cholesterol|"
                 r"triglycer|bilirubin|an(a)?emi|\bpanel\b"),
    Domain.RADIOLOGY: (r"x-?ray|\bcxr\b|\bct\b|\bmri\b|ultrasound|\busg\b|sonograph|\bscans?\b|imaging|radiolog|"
                       r"mammo|doppler|nodule|lesion|opacit|consolidation|fracture|radiograph|\bfilm\b|contrast"),
    Domain.ALLERGY: (r"allerg|prescrib|cross-?reactiv|anaphyla|contraindicat|hypersensitiv|reaction to|"
                     r"safe to (give|take|use)|can (i|he|she|they|we) (take|have|eat)|drug interaction|epipen|"
                     r"adrenaline|epinephrine|intoleran"),
    Domain.PATIENT_RECORD: r"discharge|admission|medication list|current medications|medical history|history of",
}
DOMAIN_NAMES = {Domain.LAB: "lab", Domain.RADIOLOGY: "imaging", Domain.ALLERGY: "allergy",
                Domain.PATIENT_RECORD: "patient-record (discharge / history)", Domain.REFERENCE: "reference"}
PERSONAL = re.compile(r"\b(my|me|mine|i|i'm|im|i've|ive|i am|i have|i feel|i got|myself)\b", re.I)
EXPLAIN = re.compile(r"(explain|translate|write|draft).{0,40}(to|for) (the |my )?patient|patient[- ]friendly|"
                     r"plain (language|english) for", re.I)
CASE_STATUS_ASK = re.compile(
    r"\b(status|progress|stage|update[sd]?|where)\b[^.?!]{0,40}\bcase\b|\bcase\b[^.?!]{0,30}\b(status|progress|stage|now)\b|"
    r"\b(has|have|did)\b[^.?!]{0,25}\b(doctor|clinician|radiologist|coordinator|care team)\b[^.?!]{0,30}"
    r"\b(see|seen|review|reviewed|read|check|checked|look|looked|confirm|confirmed)\b|what happened (to|with) my case", re.I)
REVIEW_ASK = re.compile(
    r"\b(doctor|clinician|physician|care team|specialist)\b[^.?!]{0,30}\b(review|check|look at|see|confirm|verify)\b|"
    r"\b(review|check|look at|confirm|verify)\b[^.?!]{0,30}\bby (a|my|the) (doctor|clinician|physician)\b|"
    r"\b(open|create|raise|start|file) (a )?case\b|\b(talk|speak|consult) (to|with) (a|my|the) (doctor|clinician)\b|"
    r"\b(send|forward|share) (it|this|them|my reports?) to (a|my|the) (doctor|clinician|care team)\b", re.I)
SAFETY_ASK = re.compile(r"safe|take|give|prescrib|avoid|react|allerg|can (i|he|she|they)", re.I)


@dataclass
class RoutePlan:
    mode: str = "case"  # case | general | explain | status
    patient_key: Optional[str] = None
    patient_source: str = ""  # session | question | none
    focus: List[Domain] = field(default_factory=list)
    denied_domains: List[Domain] = field(default_factory=list)
    agents: List[AgentName] = field(default_factory=list)
    drugs: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    check_allergies: bool = False
    wants_review: bool = False  # patient explicitly asked for a doctor to look at it -> open a case
    drug_notes: List[str] = field(default_factory=list)  # web look-ups of medicines not in the knowledge graph
    blocked: str = ""
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        names = " → ".join(AGENT_LABELS[a] for a in self.agents) or "none"
        return names + "".join(f"\n- {r}" for r in self.notes)

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "patient_key": self.patient_key, "agents": [a.value for a in self.agents],
                "denied_domains": [d.value for d in self.denied_domains], "symptoms": self.symptoms, "drugs": self.drugs}

    def block(self, message: str) -> "RoutePlan":
        self.blocked = message
        return self


def detect_domains(text: str) -> List[Domain]:
    low = text.lower()
    return [d for d, rx in DOMAIN_PATTERNS.items() if re.search(rx, low)]


DENIED_HINT = {Role.RADIOLOGIST: "As a radiologist you can work with imaging reports (contrast-safety alerts still "
                                 "appear when you ask about contrast). Lab results, allergy records and full patient "
                                 "records are available to clinicians.",
               Role.CLINICIAN: "As a clinician you can work with lab results, allergies and patient records. Imaging "
                               "is interpreted by the radiologist — their read appears on the patient's case."}
PRIVACY_BLOCK = ("For privacy, patients can only ask about their own records. "
                 "I can help you with questions about your own health and results.")


class Router:
    def __init__(self, patients: PatientDirectory, kg: MedicalKnowledgeGraph):
        self.patients = patients
        self.kg = kg

    def route(self, question: str, role: Role | str, session_patient: Optional[str] = None,
              own_patient: Optional[str] = None, has_last_insight: bool = False) -> RoutePlan:
        role = Role(role)
        pol = policy(role)
        plan = RoutePlan()
        if not self._resolve_patient(plan, question, pol, session_patient, own_patient):
            return plan
        plan.drugs = self.kg.resolve_drugs(question)
        plan.allergens = self.kg.allergens_in_text(question)
        plan.symptoms = self.kg.symptoms_in_text(question)
        plan.wants_review = role == Role.PATIENT and bool(REVIEW_ASK.search(question))
        if self._is_status_question(plan, question):
            plan.mode = "status"
            plan.notes.append("Case-status question — answered from the care-team workflow (no AI call)")
            return plan
        if not self._set_domains(plan, question, pol):
            return plan
        plan.mode = self._mode(plan, question, pol)
        plan.agents = self._agents(plan, pol, has_last_insight)
        if plan.symptoms:
            plan.notes.append("Symptoms recognised: " + ", ".join(plan.symptoms))
        return plan

    def _resolve_patient(self, plan: RoutePlan, question: str, pol: RolePolicy, session_patient: Optional[str],
                         own_patient: Optional[str]) -> bool:
        """Whose records? Patients: always their own. Staff: the patient named in the question, else the selected one."""
        named = self.patients.find_in_text(question)
        if pol.own_record_only:
            if named and named != own_patient:
                plan.block(PRIVACY_BLOCK)
                return False
            plan.patient_key, plan.patient_source = own_patient, "session"
        elif named:
            plan.patient_key, plan.patient_source = named, "question"
        elif session_patient:
            plan.patient_key, plan.patient_source = session_patient, "session"
        else:
            plan.patient_source = "none"
        if not can_access_patient(pol.role, own_patient, plan.patient_key):
            plan.block("You do not have access to this patient's records.")
            return False
        return True

    @staticmethod
    def _is_status_question(plan: RoutePlan, question: str) -> bool:
        """"Where is my case?" — answered from the case workflow, no agents / LLM."""
        return bool(plan.patient_key and CASE_STATUS_ASK.search(question) and not plan.wants_review
                    and not (plan.symptoms or plan.drugs or plan.allergens))

    @staticmethod
    def _set_domains(plan: RoutePlan, question: str, pol: RolePolicy) -> bool:
        """Record domains the question is about, filtered by RBAC (e.g. radiologists cannot open lab reports)."""
        wanted = detect_domains(question)
        if (plan.drugs or plan.allergens) and SAFETY_ASK.search(question) and Domain.ALLERGY not in wanted:
            wanted.append(Domain.ALLERGY)
        plan.focus = [d for d in wanted if d in pol.domains]
        plan.denied_domains = [d for d in wanted if d not in pol.domains]
        plan.check_allergies = Domain.ALLERGY in plan.focus or bool(plan.drugs) or pol.role == Role.PATIENT
        if not plan.denied_domains:
            return True
        names = ", ".join(DOMAIN_NAMES[d] for d in plan.denied_domains)
        plan.notes.append(f"Not permitted for the {pol.label} role: {names} records")
        if plan.focus:
            return True
        plan.block(f"{names.capitalize()} records are not available for the {pol.label} role. "
                   f"{DENIED_HINT.get(pol.role, '')}".strip())
        return False

    @staticmethod
    def _mode(plan: RoutePlan, question: str, pol: RolePolicy) -> str:
        if pol.role == Role.PATIENT:
            if PERSONAL.search(question) or plan.symptoms or plan.wants_review:
                return "case"
            plan.notes.append("General health question — answered with reference information (no personal records used)")
            return "general"
        if EXPLAIN.search(question) and AgentName.COMMUNICATION in pol.agents:
            return "explain"
        return "case"

    @staticmethod
    def _agents(plan: RoutePlan, pol: RolePolicy, has_last_insight: bool) -> List[AgentName]:
        if plan.mode == "general":
            return [AgentName.COMMUNICATION]
        if plan.mode == "explain" and has_last_insight:
            plan.notes.append("Re-using the last clinical insight for a patient-friendly explanation")
            return [AgentName.COMMUNICATION]
        agents = [a for a in AGENT_ORDER if a in pol.agents]
        if pol.role != Role.PATIENT and plan.mode != "explain":
            agents = [a for a in agents if a != AgentName.COMMUNICATION]
        return agents
