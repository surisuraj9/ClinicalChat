"""💬 Patient Communication Agent (AI, one streamed call): plain-language answer for the patient, written from the
clinical findings (or, for a general question, from reference material and trusted web sources)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from clinicalchat.agents.llm.base import LLMAgent
from clinicalchat.agents.llm.prompts import COMMUNICATION_PROMPT, COMMUNICATION_SYSTEM, GENERAL_SYSTEM
from clinicalchat.domain.roles import AgentName


class CommunicationOutput(BaseModel):
    message: str = Field(description="Plain-language answer to the patient's question (50-110 words, short sentences, no jargon)")
    what_it_means: List[str] = Field(default_factory=list, description="2-3 simple bullet points, under 15 words each")
    next_steps: List[str] = Field(default_factory=list, description="2-3 things the patient can do next (no medicine doses)")
    seek_help_if: List[str] = Field(default_factory=list, description="1-3 warning signs that need urgent care")
    questions_for_doctor: List[str] = Field(default_factory=list, description="1-2 useful questions to ask their doctor")


class PatientCommunicationAgent(LLMAgent):
    name = AgentName.COMMUNICATION
    output_model = CommunicationOutput
    text_field = "message"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, COMMUNICATION_PROMPT)

    def stream(self, *, question: str, patient: str, risk: str, insight: str, general: bool = False,
               config: Optional[dict] = None) -> Iterator[Tuple[str, Any]]:
        return self._stream({"system": GENERAL_SYSTEM if general else COMMUNICATION_SYSTEM, "question": question,
                             "patient": patient, "risk": risk, "insight": insight}, config)


def bullets(title: str, items) -> str:
    items = [i for i in (items or []) if str(i).strip()]
    return f"\n\n**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""


def communication_markdown(o: Dict[str, Any]) -> str:
    md = str(o.get("message", "")).strip()
    md += bullets("What this may mean", o.get("what_it_means")) + bullets("What you can do next", o.get("next_steps"))
    md += bullets("Get urgent help if", o.get("seek_help_if")) + bullets("Questions to ask your doctor", o.get("questions_for_doctor"))
    return md
