"""💡 Insight Generation Agent (AI, one call): consolidates the deterministic findings and the records into a
structured clinical insight with next steps and an honest confidence. Used for staff and, in the background,
for the care team on patient cases."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from clinicalchat.agents.llm.base import AgentResult, LLMAgent
from clinicalchat.agents.llm.prompts import INSIGHT_PROMPT, INSIGHT_SYSTEM
from clinicalchat.domain.roles import AgentName


class InsightOutput(BaseModel):
    answer: str = Field(description="Direct answer to the question for the stated audience (markdown, 80-220 words)")
    key_findings: List[str] = Field(default_factory=list, description="Most important findings, at most 6, one short line each")
    possible_conditions: List[str] = Field(default_factory=list, description="Conditions the findings may suggest (not a diagnosis), at most 4")
    next_steps: List[str] = Field(default_factory=list, description="Recommended follow-up tests / actions, at most 5")
    additional_data_needed: List[str] = Field(default_factory=list, description="Information that is missing to be confident, at most 3")
    confidence: float = Field(default=0.7, description="Your confidence in this insight, number between 0 and 1")
    sources: List[str] = Field(default_factory=list, description="File names of the records used, at most 5")


class InsightGenerationAgent(LLMAgent):
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
