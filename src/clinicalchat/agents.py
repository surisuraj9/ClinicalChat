"""LLM agents (LangChain LCEL, exactly one LLM call each, cached):

* 💡 Insight Generation Agent — consolidates the deterministic Extraction / Analysis / Risk outputs and the records into
  a structured, actionable clinical insight (+ confidence). Low confidence / missing data / contradictions ->
  "Review Required".
* 💬 Patient Communication Agent — turns the clinical insight into a plain-language explanation for the patient (or
  answers a general health question); a deterministic guard checks doses, definitive diagnoses and over-reassurance.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_partial_json
from pydantic import BaseModel, Field, ValidationError

from clinicalchat.rbac import AGENT_LABELS, AgentName

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- structured outputs (Pydantic v2)
class InsightOutput(BaseModel):
    answer: str = Field(description="Direct answer to the question for the stated audience (markdown, 80-220 words)")
    key_findings: List[str] = Field(default_factory=list, description="Most important findings, one short line each")
    possible_conditions: List[str] = Field(default_factory=list, description="Conditions the findings may suggest (not a diagnosis)")
    next_steps: List[str] = Field(default_factory=list, description="Recommended follow-up tests / actions")
    additional_data_needed: List[str] = Field(default_factory=list, description="Information that is missing to be confident")
    confidence: float = Field(default=0.7, description="Your confidence in this insight, number between 0 and 1")
    sources: List[str] = Field(default_factory=list, description="File names of the records used")


class CommunicationOutput(BaseModel):
    message: str = Field(description="Plain-language answer to the patient's question (50-110 words, short sentences, no jargon)")
    what_it_means: List[str] = Field(default_factory=list, description="2-3 simple bullet points, under 15 words each")
    next_steps: List[str] = Field(default_factory=list, description="2-3 things the patient can do next (no medicine doses)")
    seek_help_if: List[str] = Field(default_factory=list, description="1-3 warning signs that need urgent care")
    questions_for_doctor: List[str] = Field(default_factory=list, description="1-2 useful questions to ask their doctor")


INSIGHT_SYSTEM = (
    "You are the INSIGHT GENERATION agent of a clinical decision-support system. Upstream agents have already run: "
    "DATA EXTRACTION (which records were found, their quality, missing data), CLINICAL ANALYSIS (lab values vs "
    "reference ranges, imaging findings, allergy/drug safety, symptom differential, inconsistencies) and RISK "
    "DETECTION (rule-based risk level). Consolidate their outputs and the records into clear, actionable insight "
    "that answers the question. Recommend next steps (tests, referrals, follow-up). State what additional data "
    "would be needed and give an honest confidence.")
COMMUNICATION_SYSTEM = (
    "You are the PATIENT COMMUNICATION agent. Translate the clinical insight below into a short, warm, plain-language "
    "explanation for the patient. Keep it informational, never diagnostic: say what the findings MAY mean, not what the "
    "patient HAS. Never give medicine names with doses, never tell them to start/stop medicines, never be "
    "over-reassuring when results are abnormal. Always encourage discussing with their doctor. Follow the risk "
    "guidance exactly (if it says emergency, say so first).")
GENERAL_SYSTEM = (
    "You are the PATIENT COMMUNICATION agent answering a general health question from a patient. Give accurate, "
    "general, educational information in plain language using the reference material when relevant. Do not diagnose, "
    "do not give medicine doses. If the patient's documented allergies are relevant, mention them. If the question "
    "is not about health or medicine, politely say you can only help with health questions. Encourage seeing a "
    "doctor for personal medical advice.")

GROUND_RULES = """Ground rules:
- Use ONLY the information provided. If something is missing, say so. Never invent values, dates, patients or reports.
- DETERMINISTIC FINDINGS and RISK were computed by validated code (reference ranges, knowledge graph, safety rules): never contradict them.
- Allergy and drug-safety information takes priority over everything else.
- You support, not replace, clinical judgement."""

INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}\n\n{rules}\n\nAUDIENCE: {audience}\n\n"
               "Reply with ONE JSON object only — no text before or after — with these keys:\n{schema}"),
    ("human", "CONVERSATION SO FAR:\n{history}\n\nQUESTION:\n{question}\n\nPATIENT PROFILE:\n{patient}\n\n"
              "DATA EXTRACTION:\n{extraction}\n\nDETERMINISTIC FINDINGS (Clinical Analysis):\n{facts}\n\n"
              "RISK (Risk Detection):\n{risk}\n\nLESSONS FROM CLINICIAN REVIEWS:\n{feedback}\n\nRECORDS:\n{context}"),
])
COMMUNICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}\n\n{rules}\n\nReply with ONE JSON object only — no text before or after — with these keys:\n{schema}"),
    ("human", "PATIENT'S QUESTION:\n{question}\n\nPATIENT:\n{patient}\n\nRISK GUIDANCE:\n{risk}\n\n"
              "CLINICAL INSIGHT / REFERENCE MATERIAL:\n{insight}"),
])


def schema_hint(model: Type[BaseModel]) -> str:
    lines = []
    for name, f in model.model_fields.items():
        ann = str(f.annotation)
        kind = "list of strings" if "List" in ann else "number" if "float" in ann else "string"
        lines.append(f'- "{name}" ({kind}){": " + f.description if f.description else ""}')
    return "\n".join(lines)


def parse_output(raw: str, model: Type[BaseModel], text_field: str) -> tuple[BaseModel, bool]:
    """Lenient JSON extraction; falls back to the raw text (never a second LLM call)."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                for k, v in list(data.items()):
                    if k not in model.model_fields:
                        continue
                    ann = str(model.model_fields[k].annotation)
                    if "List" in ann:
                        if isinstance(v, str):
                            data[k] = [v] if v else []
                        elif isinstance(v, list):
                            data[k] = [x if isinstance(x, str) else " — ".join(map(str, x.values())) if isinstance(x, dict)
                                       else str(x) for x in v]
                    elif "float" in ann:
                        try:
                            data[k] = max(0.0, min(1.0, float(str(v).strip("%")) / (100 if float(str(v).strip("%")) > 1 else 1)))
                        except ValueError:
                            data.pop(k)
                return model.model_validate(data), True
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("Could not parse agent JSON: %s", str(exc)[:200])
    return model(**{text_field: text.strip("`") or "No answer was produced."}), False


class AgentResult(BaseModel):
    agent: AgentName
    label: str
    output: Optional[Dict[str, Any]] = None
    parsed: bool = True
    error: str = ""
    latency_ms: int = 0
    cache_hit: bool = False


class _LLMAgent:
    name: AgentName
    output_model: Type[BaseModel]
    text_field: str

    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.llm = llm
        self.label = AGENT_LABELS[self.name]
        self.chain = prompt | llm  # LCEL; parsing happens after so cache metadata stays visible

    def _invoke(self, variables: Dict[str, Any], config: Optional[dict]) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(agent=self.name, label=self.label)
        try:
            message = self.chain.invoke({"rules": GROUND_RULES, "schema": schema_hint(self.output_model), **variables},
                                        config={**(config or {}), "run_name": self.label,
                                                "metadata": {**(config or {}).get("metadata", {}), "agent": self.name.value}})
            result.cache_hit = bool((getattr(message, "response_metadata", None) or {}).get("cache_hit"))
            output, ok = parse_output(StrOutputParser().invoke(message), self.output_model, self.text_field)
            result.output, result.parsed = output.model_dump(), ok
        except Exception as exc:  # a backend failure must not crash the chat
            log.exception("%s failed", self.name.value)
            result.error = str(exc)[:400]
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result


    def _stream(self, variables: Dict[str, Any], config: Optional[dict]) -> Iterator[Tuple[str, Any]]:
        """Like `_invoke`, but yields ("partial", dict) while the model is still writing (OpenAI streams token by
        token; backends without streaming yield one final chunk), then ("done", AgentResult)."""
        start = time.perf_counter()
        result = AgentResult(agent=self.name, label=self.label)
        text, shown = "", 0
        try:
            for chunk in self.chain.stream({"rules": GROUND_RULES, "schema": schema_hint(self.output_model), **variables},
                                           config={**(config or {}), "run_name": self.label,
                                                   "metadata": {**(config or {}).get("metadata", {}),
                                                                "agent": self.name.value}}):
                if not text:
                    result.cache_hit = bool((getattr(chunk, "response_metadata", None) or {}).get("cache_hit"))
                text += chunk.content if isinstance(chunk.content, str) else StrOutputParser().invoke(chunk)
                if len(text) - shown >= 24:  # re-render roughly every few words
                    partial = parse_partial_json(_FENCE.sub("", text).strip() or "{}")
                    if isinstance(partial, dict) and partial:
                        shown = len(text)
                        yield "partial", partial
            output, ok = parse_output(text, self.output_model, self.text_field)
            result.output, result.parsed = output.model_dump(), ok
        except Exception as exc:  # a backend failure must not crash the chat
            log.exception("%s failed", self.name.value)
            result.error = str(exc)[:400]
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        yield "done", result


_FENCE = re.compile(r"^```(?:json)?|```$", re.M)


class InsightGenerationAgent(_LLMAgent):
    name = AgentName.INSIGHT
    output_model = InsightOutput
    text_field = "answer"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, INSIGHT_PROMPT)

    def run(self, *, question: str, audience: str, patient: str, extraction: str, facts: str, risk: str, feedback: str,
            context: str, history: str, config: Optional[dict] = None) -> AgentResult:
        return self._invoke({"system": INSIGHT_SYSTEM, "audience": audience, "question": question, "patient": patient,
                             "extraction": extraction, "facts": facts, "risk": risk, "feedback": feedback or "None.",
                             "context": context, "history": history or "None."}, config)


class PatientCommunicationAgent(_LLMAgent):
    name = AgentName.COMMUNICATION
    output_model = CommunicationOutput
    text_field = "message"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, COMMUNICATION_PROMPT)

    def run(self, *, question: str, patient: str, risk: str, insight: str, general: bool = False,
            config: Optional[dict] = None) -> AgentResult:
        return self._invoke(self._vars(question, patient, risk, insight, general), config)

    def stream(self, *, question: str, patient: str, risk: str, insight: str, general: bool = False,
               config: Optional[dict] = None) -> Iterator[Tuple[str, Any]]:
        return self._stream(self._vars(question, patient, risk, insight, general), config)

    @staticmethod
    def _vars(question: str, patient: str, risk: str, insight: str, general: bool) -> Dict[str, Any]:
        return {"system": GENERAL_SYSTEM if general else COMMUNICATION_SYSTEM, "question": question,
                "patient": patient, "risk": risk, "insight": insight}


# ---------------------------------------------------------------- patient-text guard (deterministic)
# a lab concentration ("15 g/dL", "126 mg per decilitre") is a result, not a dose
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|units?|iu)\b"
                   r"(?!\s*(?:/|per\s+)(?:dl|l|ml|deciliter|decilitre|liter|litre|cu\s?mm|mm3|ul|µl)\b)"
                   r"(?:\s*(?:once|twice|daily|a day|per day|bd|od|tds))?", re.I)
_DEFINITIVE = re.compile(r"\byou (?:definitely |certainly )?(?:have|are suffering from|are diagnosed with) "
                         r"(?:cancer|a tumou?r|leukaemia|leukemia|kidney failure|heart failure|a stroke|diabetes|hiv|pneumonia)\b", re.I)
_SELF_MED = re.compile(r"\b(stop|start|double|increase|decrease) (?:taking )?(?:your |the )?(?:medicine|medication|tablets?|insulin|dose)\b", re.I)
_REASSURE = re.compile(r"\b(nothing to worry about|completely normal|no need to see (?:a|your) doctor)\b", re.I)


def check_patient_text(parts: Iterable[str], has_abnormal: bool) -> List[str]:
    text = "\n".join(p for p in parts if p)
    issues: List[str] = []
    if _DOSE.search(text):
        issues.append("Contains medication doses — patient messages must not include dosing")
    if _DEFINITIVE.search(text):
        issues.append("States a definitive diagnosis — must stay informational until a clinician confirms")
    if _SELF_MED.search(text):
        issues.append("Suggests changing medication without a clinician")
    if has_abnormal and _REASSURE.search(text):
        issues.append("Over-reassuring wording although abnormal findings are present")
    if "doctor" not in text.lower() and "clinician" not in text.lower():
        issues.append("Does not direct the patient to discuss with their doctor")
    return issues


def communication_markdown(o: Dict[str, Any]) -> str:
    md = str(o.get("message", "")).strip()
    md += _bullets("What this may mean", o.get("what_it_means")) + _bullets("What you can do next", o.get("next_steps"))
    md += _bullets("Get urgent help if", o.get("seek_help_if")) + _bullets("Questions to ask your doctor", o.get("questions_for_doctor"))
    return md


def _bullets(title: str, items) -> str:
    items = [i for i in (items or []) if str(i).strip()]
    return f"\n\n**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""
