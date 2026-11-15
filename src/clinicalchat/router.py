"""Keyword router (no LLM): decides the pipeline mode, the record domains a question is about (with RBAC), the patient it
concerns and the symptoms / drugs / allergens it mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from clinicalchat.engine import ClinicalEngine
from clinicalchat.models import Domain
from clinicalchat.rbac import AGENT_LABELS, AGENT_ORDER, AgentName, Role, can_access_patient, policy

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
    blocked: str = ""
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        names = " → ".join(AGENT_LABELS[a] for a in self.agents) or "none"
        return names + "".join(f"\n- {r}" for r in self.notes)


def detect_domains(text: str) -> List[Domain]:
    low = text.lower()
    return [d for d, rx in DOMAIN_PATTERNS.items() if re.search(rx, low)]


class Router:
    def __init__(self, engine: ClinicalEngine):
        self.engine = engine

    def route(self, question: str, role: Role | str, session_patient: Optional[str] = None,
              own_patient: Optional[str] = None, has_last_insight: bool = False) -> RoutePlan:
        role = Role(role)
        pol = policy(role)
        plan = RoutePlan()
        kg = self.engine.kg

        # ---- patient resolution + data-access check
        named = self.engine.find_patient_in_text(question)
        if pol.own_record_only:
            if named and named != own_patient:
                plan.blocked = ("For privacy, patients can only ask about their own records. "
                                "I can help you with questions about your own health and results.")
                return plan
            plan.patient_key, plan.patient_source = own_patient, "session"
        elif named:
            plan.patient_key, plan.patient_source = named, "question"
        elif session_patient:
            plan.patient_key, plan.patient_source = session_patient, "session"
        else:
            plan.patient_source = "none"
        if not can_access_patient(role, own_patient, plan.patient_key):
            plan.blocked = "You do not have access to this patient's records."
            return plan

        # ---- entities (deterministic, knowledge graph)
        plan.drugs = kg.resolve_drugs(question)
        plan.allergens = kg.allergens_in_text(question)
        plan.symptoms = kg.symptoms_in_text(question)

        # ---- "where is my case?" — answered from the case workflow, no agents / LLM
        if plan.patient_key and CASE_STATUS_ASK.search(question) and not (plan.symptoms or plan.drugs or plan.allergens):
            plan.mode = "status"
            plan.notes.append("Case-status question — answered from the care-team workflow (no AI call)")
            return plan

        # ---- record domains (RBAC: e.g. radiologists cannot open lab reports)
        wanted = detect_domains(question)
        if (plan.drugs or plan.allergens) and SAFETY_ASK.search(question) and Domain.ALLERGY not in wanted:
            wanted.append(Domain.ALLERGY)
        plan.focus = [d for d in wanted if d in pol.domains]
        plan.denied_domains = [d for d in wanted if d not in pol.domains]
        plan.check_allergies = Domain.ALLERGY in plan.focus or bool(plan.drugs) or role == Role.PATIENT
        if plan.denied_domains:
            names = ", ".join(DOMAIN_NAMES[d] for d in plan.denied_domains)
            plan.notes.append(f"Not permitted for the {pol.label} role: {names} records")
            if not plan.focus:
                plan.blocked = self._denied_message(role, names)
                return plan

        # ---- mode
        personal = bool(PERSONAL.search(question))
        if role == Role.PATIENT:
            plan.mode = "case" if (personal or plan.symptoms) else "general"
        elif EXPLAIN.search(question) and AgentName.COMMUNICATION in pol.agents:
            plan.mode = "explain"
        else:
            plan.mode = "case"
        if plan.mode == "general":
            plan.notes.append("General health question — answered with reference information (no personal records used)")

        if plan.mode == "general":
            plan.agents = [AgentName.COMMUNICATION] if role == Role.PATIENT else [AgentName.INSIGHT]
        elif plan.mode == "explain" and has_last_insight:
            plan.agents = [AgentName.COMMUNICATION]
            plan.notes.append("Re-using the last clinical insight for a patient-friendly explanation")
        else:
            plan.agents = [a for a in AGENT_ORDER if a in pol.agents]
            if role != Role.PATIENT and plan.mode != "explain":
                plan.agents = [a for a in plan.agents if a != AgentName.COMMUNICATION]
        if plan.symptoms:
            plan.notes.append("Symptoms recognised: " + ", ".join(plan.symptoms))
        return plan

    @staticmethod
    def _denied_message(role: Role, names: str) -> str:
        hints = {
            Role.RADIOLOGIST: "As a radiologist you can work with imaging reports and allergy / contrast safety. "
                              "Lab interpretation and full patient records are available to clinicians.",
            Role.COORDINATOR: "",
            Role.CLINICIAN: "",
            Role.PATIENT: "",
        }
        return f"{names.capitalize()} records are not available for the {policy(role).label} role. {hints[role]}".strip()
