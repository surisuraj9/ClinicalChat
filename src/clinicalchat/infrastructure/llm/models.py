"""OpenAI chat models: a strong model for the clinician-facing Insight agent, a fast one for patient answers.

Without ``OPENAI_API_KEY`` the app still starts: a placeholder model raises a clear error on use, and every
deterministic feature (lab flags, safety checks, risk rules, care-team workflow) keeps working.
"""

from __future__ import annotations

from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

from clinicalchat.config import Settings
from clinicalchat.infrastructure.llm.cache import SQLiteLLMCache

INSIGHT, FAST, SPECIALIST = "insight", "fast", "specialist"
NO_KEY = "No LLM configured: set OPENAI_API_KEY in .env."


class UnavailableChatModel(BaseChatModel):
    reason: str = NO_KEY

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError(self.reason)


def install_cache(settings: Settings) -> None:
    """Identical prompts are answered from SQLite instead of calling OpenAI again (process-wide LangChain cache)."""
    if settings.llm_cache:
        set_llm_cache(SQLiteLLMCache(settings.cache_dir / "llm_cache.sqlite"))


def create_chat_model(settings: Settings, purpose: str = INSIGHT) -> BaseChatModel:
    if not settings.openai_api_key:
        return UnavailableChatModel()
    from langchain_openai import ChatOpenAI

    model, max_tokens = {
        FAST: (settings.openai_fast_model, settings.communication_max_tokens),
        SPECIALIST: (settings.openai_specialist_model, settings.specialist_max_tokens),
    }.get(purpose, (settings.openai_model, settings.insight_max_tokens))
    return ChatOpenAI(model=model, temperature=0.0, api_key=settings.openai_api_key, timeout=settings.openai_timeout,
                      max_retries=settings.llm_retries, stream_usage=True, max_tokens=max_tokens,
                      model_kwargs={"response_format": {"type": "json_object"}})


def warm_up(*models: BaseChatModel) -> None:
    """Open the HTTPS connection to OpenAI before the first question (~1 s saved). Costs no tokens."""
    for model in models:
        try:
            model.root_client.models.retrieve(model.model_name)
        except Exception:  # best-effort (no key, offline, placeholder model)
            pass


def describe_llm(settings: Settings) -> dict:
    if not settings.openai_api_key:
        return {"provider": "unavailable", "error": NO_KEY}
    return {"provider": "openai", "model": settings.openai_model, "fast_model": settings.openai_fast_model,
            "specialist_model": settings.openai_specialist_model if settings.specialists_use_llm else "off (rule-based)",
            "cache": settings.llm_cache}
