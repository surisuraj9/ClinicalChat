"""Base class of the AI agents: one LangChain call (prompt | model), timed, traced, parsed and never crashing the chat."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_partial_json
from pydantic import BaseModel

from clinicalchat.agents.llm.output import FENCE, parse_output, schema_hint
from clinicalchat.agents.llm.prompts import GROUND_RULES
from clinicalchat.domain.roles import AGENT_LABELS, AgentName

log = logging.getLogger(__name__)
STREAM_EVERY_CHARS = 24  # re-render the partial answer roughly every few words


class AgentResult(BaseModel):
    agent: AgentName
    label: str
    output: Optional[Dict[str, Any]] = None
    parsed: bool = True
    error: str = ""
    latency_ms: int = 0
    cache_hit: bool = False

    def timing(self) -> str:
        if self.error:
            return self.error[:160]
        return "cached answer" if self.cache_hit else f"{self.latency_ms / 1000:.1f}s"


def _hit_length_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "LengthFinishReasonError" or "length limit was reached" in str(exc)


def _partial_json(text: str) -> Any:
    return parse_partial_json(FENCE.sub("", text).strip() or "{}")


class LLMAgent:
    name: AgentName
    output_model: Type[BaseModel]
    text_field: str

    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.llm = llm
        self.label = AGENT_LABELS[self.name]
        self.prompt = prompt
        self.chain = prompt | llm  # LCEL; parsing happens after so cache metadata stays visible

    def _inputs(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        return {"rules": GROUND_RULES, "schema": schema_hint(self.output_model), **variables}

    def _config(self, config: Optional[dict]) -> dict:
        return {**(config or {}), "run_name": self.label,
                "metadata": {**(config or {}).get("metadata", {}), "agent": self.name.value}}

    def _invoke(self, variables: Dict[str, Any], config: Optional[dict]) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(agent=self.name, label=self.label)
        try:
            message = self._call(self._inputs(variables), self._config(config))
            result.cache_hit = bool((getattr(message, "response_metadata", None) or {}).get("cache_hit"))
            output, ok = self._parse(StrOutputParser().invoke(message))
            result.output, result.parsed = output.model_dump(), ok
        except Exception as exc:  # a backend failure must not crash the chat
            log.exception("%s failed", self.name.value)
            result.error = str(exc)[:400]
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    def _call(self, inputs: Dict[str, Any], cfg: dict) -> Any:
        try:
            return self.chain.invoke(inputs, config=cfg)
        except Exception as exc:
            cap = getattr(self.llm, "max_tokens", None)
            if not (cap and _hit_length_limit(exc)):
                raise
            # a long answer (e.g. many lab values) ran out of room: retry once with double the output budget
            log.warning("%s hit the %d-token output cap; retrying with %d", self.name.value, cap, cap * 2)
            return (self.prompt | self.llm.bind(max_tokens=cap * 2)).invoke(inputs, config=cfg)

    def _stream(self, variables: Dict[str, Any], config: Optional[dict]) -> Iterator[Tuple[str, Any]]:
        """Like ``_invoke``, but yields ("partial", dict) while the model is still writing, then ("done", AgentResult)."""
        start = time.perf_counter()
        result = AgentResult(agent=self.name, label=self.label)
        text, shown = "", 0
        try:
            for chunk in self.chain.stream(self._inputs(variables), config=self._config(config)):
                if not text:
                    result.cache_hit = bool((getattr(chunk, "response_metadata", None) or {}).get("cache_hit"))
                text += chunk.content if isinstance(chunk.content, str) else StrOutputParser().invoke(chunk)
                if len(text) - shown >= STREAM_EVERY_CHARS:
                    partial = _partial_json(text)
                    if isinstance(partial, dict) and partial:
                        shown = len(text)
                        yield "partial", partial
            output, ok = self._final(text)
            result.output, result.parsed = output.model_dump(), ok
        except Exception as exc:  # a backend failure must not crash the chat
            log.exception("%s failed", self.name.value)
            result.error = str(exc)[:400]
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        yield "done", result

    def _parse(self, text: str) -> Tuple[BaseModel, bool]:
        """Reply text -> (output model, parsed_ok). Agents with nested outputs override this."""
        return parse_output(text, self.output_model, self.text_field)

    def _final(self, text: str) -> Tuple[BaseModel, bool]:
        output, ok = self._parse(text)
        if not ok:  # cut off by the output cap: keep the complete fields that were written
            partial = _partial_json(text)
            if isinstance(partial, dict) and partial.get(self.text_field):
                output, ok = parse_output(json.dumps(partial), self.output_model, self.text_field)
        return output, ok
