"""Specialist agents: one LCEL chain (prompt | llm | parser) and exactly one LLM call per agent."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from clinicalchat.rbac import AGENT_LABELS, AgentName

log = logging.getLogger(__name__)

URGENCY = "ROUTINE | SOON | URGENT | EMERGENCY"


# ---------------------------------------------------------------- structured outputs (Pydantic v2)
class AgentOutput(BaseModel):
    answer: str = Field(description="Direct answer to the question for the stated audience (markdown, 80-220 words)")
    key_findings: List[str] = Field(default_factory=list, description="Most important findings, each one short line")
    recommendations: List[str] = Field(default_factory=list, description="Recommended follow-up tests / next steps")
    urgency: str = Field(default="ROUTINE", description=URGENCY)
    sources: List[str] = Field(default_factory=list, description="File names of records used")


class LabOutput(AgentOutput):
    abnormal_values: List[str] = Field(default_factory=list, description="'<parameter>: <value> (<ref range>) HIGH/LOW'")
    possible_conditions: List[str] = Field(default_factory=list, description="Conditions the pattern may suggest")


class RadiologyOutput(AgentOutput):
    study: str = Field(default="", description="Modality and body region, e.g. 'CT Chest'")
    impression: str = ""
    differential_diagnosis: List[str] = Field(default_factory=list)
    critical_findings: List[str] = Field(default_factory=list, description="Findings that need urgent communication")


class AllergyOutput(AgentOutput):
    risk_level: str = Field(default="NONE", description="NONE | LOW | MODERATE | HIGH | CRITICAL")
    cross_reactivities: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list, description="Drugs/foods/substances to avoid")
    safe_alternatives: List[str] = Field(default_factory=list)
    emergency_protocol: str = Field(default="", description="What to do if a severe reaction occurs")


class CoordinatorOutput(AgentOutput):
    care_status: str = Field(default="", description="One-paragraph current status across specialties")
    pending_items: List[str] = Field(default_factory=list, description="Outstanding results, tests or decisions")
    specialists: List[str] = Field(default_factory=list, description="Specialties/teams that should be involved")
    handoff_notes: List[str] = Field(default_factory=list, description="Critical points for the receiving team (SBAR style)")


SYSTEM_PROMPTS: Dict[AgentName, str] = {
    AgentName.LAB: ("You are the LAB INTERPRETER agent. You interpret CBC, liver (LFT), kidney (KFT), thyroid, blood "
                    "sugar, cardiac/lipid and BMI results: flag abnormal values against the given reference ranges, "
                    "explain clinical significance, identify patterns across tests and recommend follow-up tests."),
    AgentName.RADIOLOGY: ("You are the RADIOLOGY ANALYZER agent. You review X-ray, CT, MRI and ultrasound reports: "
                          "summarise findings by region, give the impression, a differential diagnosis, flag critical "
                          "findings and recommend further imaging or correlation."),
    AgentName.ALLERGY: ("You are the ALLERGY SAFETY agent. You check medications/foods against the patient's documented "
                        "allergies, explain cross-reactivity, severity and contraindications, propose safe alternatives "
                        "and state the emergency protocol (e.g. anaphylaxis: IM adrenaline 0.5 mg, call emergency "
                        "services). Patient safety overrides everything else."),
    AgentName.COORDINATOR: ("You are the CARE COORDINATOR agent. You give a cross-specialty view of the patient (labs, "
                            "imaging, allergies, medications, conditions): current care status, pending items, which "
                            "specialists to involve, follow-up scheduling and a concise handoff summary."),
}
OUTPUT_MODELS: Dict[AgentName, Type[AgentOutput]] = {
    AgentName.LAB: LabOutput, AgentName.RADIOLOGY: RadiologyOutput,
    AgentName.ALLERGY: AllergyOutput, AgentName.COORDINATOR: CoordinatorOutput,
}

GROUND_RULES = """Ground rules:
- Use ONLY the patient profile, deterministic facts and records below. If information is missing, say so. Never invent values, dates or patients.
- DETERMINISTIC FACTS were computed by validated code (reference-range flags, knowledge-graph allergy checks): never contradict them.
- Allergy and drug-safety information takes priority over everything else.
- You support, not replace, clinical judgement."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}\n\n{rules}\n\nAUDIENCE: {audience}\n\n"
               "Reply with ONE JSON object only — no text before or after — with these keys:\n{schema}"),
    ("human", "CONVERSATION SO FAR:\n{history}\n\nQUESTION:\n{question}\n\nPATIENT PROFILE:\n{patient}\n\n"
              "DETERMINISTIC FACTS:\n{facts}\n\nRECORDS:\n{context}"),
])


def schema_hint(model: Type[BaseModel]) -> str:
    lines = []
    for name, f in model.model_fields.items():
        kind = "list of strings" if "List" in str(f.annotation) else "string"
        lines.append(f'- "{name}" ({kind}){": " + f.description if f.description else ""}')
    return "\n".join(lines)


def parse_output(raw: str, model: Type[AgentOutput]) -> tuple[AgentOutput, bool]:
    """Lenient JSON extraction; falls back to using the raw text as the answer (never a second LLM call)."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                for k, v in list(data.items()):
                    if k in model.model_fields and "List" in str(model.model_fields[k].annotation) and isinstance(v, str):
                        data[k] = [v] if v else []
                    elif isinstance(v, list) and k in model.model_fields and "List" in str(model.model_fields[k].annotation):
                        data[k] = [x if isinstance(x, str) else json.dumps(x) if not isinstance(x, dict) else
                                   " — ".join(str(y) for y in x.values()) for x in v]
                return model.model_validate(data), True
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("Could not parse agent JSON: %s", str(exc)[:200])
    return model(answer=text.strip("`") or "No answer was produced."), False


class AgentResult(BaseModel):
    agent: AgentName
    label: str
    output: Optional[Dict[str, Any]] = None
    parsed: bool = True
    error: str = ""
    latency_ms: int = 0
    cache_hit: bool = False
    sources: List[str] = Field(default_factory=list)


class SpecialistAgent:
    def __init__(self, name: AgentName, llm: BaseChatModel):
        self.name = name
        self.label = AGENT_LABELS[name]
        self.output_model = OUTPUT_MODELS[name]
        self.llm = llm
        self.chain = PROMPT | llm  # LCEL; StrOutputParser applied after so cache metadata stays visible

    def run(self, *, question: str, audience: str, patient: str, facts: str, context: str, history: str,
            sources: List[str], config: Optional[dict] = None) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(agent=self.name, label=self.label, sources=sources)
        try:
            message = self.chain.invoke({
                "system": SYSTEM_PROMPTS[self.name], "rules": GROUND_RULES, "audience": audience,
                "schema": schema_hint(self.output_model), "history": history or "None.", "question": question,
                "patient": patient, "facts": facts or "None.", "context": context,
            }, config={**(config or {}), "run_name": self.label, "metadata": {"agent": self.name.value}})
            result.cache_hit = bool((getattr(message, "response_metadata", None) or {}).get("cache_hit"))
            output, ok = parse_output(StrOutputParser().invoke(message), self.output_model)
            result.output, result.parsed = output.model_dump(), ok
            if not output.sources:
                result.output["sources"] = sources
        except Exception as exc:  # backend failure must not crash the chat
            log.exception("%s failed", self.name.value)
            result.error = str(exc)[:400]
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result
