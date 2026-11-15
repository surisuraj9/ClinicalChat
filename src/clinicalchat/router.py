"""Keyword router (no LLM): picks agents for a question, applies RBAC and resolves which patient it is about."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from clinicalchat.engine import ClinicalEngine
from clinicalchat.models import Domain
from clinicalchat.rbac import AGENT_LABELS, AgentName, Role, can_access_patient, can_use, policy

INTENT_PATTERNS: Dict[AgentName, str] = {
    AgentName.LAB: (r"\blabs?\b|blood (test|work|report)|\bcbc\b|blood count|h(a)?emoglobin|\bwbc\b|platelet|"
                    r"metabolic|\bcmp\b|\bbmp\b|electrolyte|liver function|\blft\b|kidney function|renal|\bkft\b|\brft\b|"
                    r"creatinine|thyroid|\btsh\b|\bt3\b|\bt4\b|sugar|glucose|hba1c|diabet|lipid|cholesterol|"
                    r"triglycer|bilirubin|\bbmi\b|an(a)?emi|panel"),
    AgentName.RADIOLOGY: (r"x-?ray|\bcxr\b|\bct\b|\bmri\b|ultrasound|\busg\b|sonograph|scan\b|scans\b|imaging|radiolog|"
                          r"mammo|doppler|nodule|lesion|opacit|consolidation|fracture|radiograph|\bfilm\b|contrast"),
    AgentName.ALLERGY: (r"allerg|prescrib|cross-?reactiv|anaphyla|contraindicat|hypersensitiv|reaction to|"
                        r"safe to (give|take|use)|can (i|he|she|they|we) (take|have|eat)|drug interaction|epipen|"
                        r"adrenaline|epinephrine|intoleran"),
    AgentName.COORDINATOR: (r"hand-?off|hand ?over|care (status|plan|coordination|summary)|coordinat|discharge|"
                            r"referr|cross-?specialt|appointment|schedul|overall status|next steps for (the )?care|"
                            r"team update|status update"),
}
SUMMARY_PATTERN = r"summar(y|ise|ize)|overview"
DOMAIN_TO_AGENT = {Domain.LAB: AgentName.LAB, Domain.RADIOLOGY: AgentName.RADIOLOGY, Domain.ALLERGY: AgentName.ALLERGY}


@dataclass
class RoutePlan:
    agents: List[AgentName] = field(default_factory=list)
    denied: List[AgentName] = field(default_factory=list)
    patient_key: Optional[str] = None
    patient_source: str = ""  # session | question | none
    drugs: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    blocked: str = ""
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        names = ", ".join(AGENT_LABELS[a] for a in self.agents) or "none"
        return names + "".join(f"\n- {r}" for r in self.notes)


def detect_intents(text: str) -> Dict[AgentName, str]:
    low = text.lower()
    found = {}
    for agent, rx in INTENT_PATTERNS.items():
        m = re.search(rx, low)
        if m:
            found[agent] = m.group(0)
    return found


class Router:
    def __init__(self, engine: ClinicalEngine):
        self.engine = engine

    def route(self, question: str, role: Role | str, session_patient: Optional[str] = None,
              own_patient: Optional[str] = None, previous_agents: Sequence[AgentName] = ()) -> RoutePlan:
        role = Role(role)
        pol = policy(role)
        plan = RoutePlan()
        kg = self.engine.kg

        # ---- patient resolution + data-access check
        named = self.engine.find_patient_in_text(question)
        if pol.own_record_only:
            if named and named != own_patient:
                plan.blocked = ("For privacy, patients can only ask about their own records. "
                                "I can help you with questions about your own results.")
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

        plan.drugs = kg.resolve_drugs(question)
        plan.allergens = kg.allergens_in_text(question)

        # ---- intent detection
        intents = detect_intents(question)
        if (plan.drugs or plan.allergens) and AgentName.ALLERGY not in intents and re.search(r"safe|take|give|prescrib|avoid|react", question.lower()):
            intents[AgentName.ALLERGY] = (plan.drugs or plan.allergens)[0]
        if not intents and re.search(SUMMARY_PATTERN, question.lower()) and can_use(role, AgentName.COORDINATOR):
            intents[AgentName.COORDINATOR] = "summary"
        wanted = list(intents)

        if not wanted and previous_agents:
            wanted = list(previous_agents)
            plan.notes.append("Follow-up question — reusing the specialists from the previous answer")
        if not wanted:
            wanted = self._fallback(question, role, plan.patient_key)
            plan.notes.append("No specific keywords — chose specialists from the available records")

        # ---- RBAC filtering (coordinators get specialist questions as a cross-specialty summary)
        selected: List[AgentName] = []
        for agent in wanted:
            if can_use(role, agent):
                selected.append(agent)
            elif role == Role.COORDINATOR and agent in (AgentName.LAB, AgentName.RADIOLOGY):
                selected.append(AgentName.COORDINATOR)
                plan.notes.append(f"{AGENT_LABELS[agent]} request answered as a care-coordination summary for your role")
            else:
                plan.denied.append(agent)
        plan.agents = list(dict.fromkeys(selected))
        plan.reasons = {a.value: intents.get(a, "context") for a in plan.agents}
        if plan.denied:
            plan.notes.append("Not available for your role: " + ", ".join(AGENT_LABELS[a] for a in plan.denied))
        if not plan.agents:
            plan.blocked = self._denied_message(role, plan.denied)
        return plan

    def _fallback(self, question: str, role: Role, patient_key: Optional[str]) -> List[AgentName]:
        if role == Role.COORDINATOR:
            return [AgentName.COORDINATOR]
        allowed = [a for a in (AgentName.LAB, AgentName.RADIOLOGY, AgentName.ALLERGY) if can_use(role, a)]
        if patient_key:
            domains = self.engine.patient_domains(patient_key)
            from_data = [DOMAIN_TO_AGENT[d] for d in domains if d in DOMAIN_TO_AGENT and DOMAIN_TO_AGENT[d] in allowed]
            if from_data:
                return from_data[:2]
        hits = self.engine.store.search(question, k=3)
        for h in hits:
            agent = DOMAIN_TO_AGENT.get(Domain(h.metadata.get("domain", "reference")))
            if agent in allowed:
                return [agent]
        if role == Role.CLINICIAN and patient_key:
            return [AgentName.COORDINATOR]
        return allowed[:1]

    @staticmethod
    def _denied_message(role: Role, denied: Sequence[AgentName]) -> str:
        names = ", ".join(AGENT_LABELS[a] for a in denied)
        hints = {
            Role.PATIENT: "Care-team handoff summaries are for clinical staff. I can explain your lab results, imaging reports or allergy safety in simple terms.",
            Role.RADIOLOGIST: "As a radiologist you can use the Radiology Analyzer and Allergy Safety (e.g. contrast safety). Lab interpretation is available to clinicians.",
            Role.COORDINATOR: "As a care coordinator you can request care-status / handoff summaries and allergy safety checks.",
            Role.CLINICIAN: "",
        }
        return f"{names} is not available for the {policy(role).label} role. {hints[role]}".strip()
