"""LLM provider factory: OpenAI (production) or GitHub Copilot CLI (local development)."""

from __future__ import annotations

import shutil
from functools import lru_cache

from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

from clinicalchat.config import Settings, get_settings
from clinicalchat.llm.cache import SQLiteLLMCache
from clinicalchat.llm.copilot_cli import CopilotCLIChatModel

_cache_installed = False


def resolve_provider(settings: Settings) -> str:
    provider = settings.llm_provider
    if provider in {"openai", "copilot", "fake"}:
        return provider
    if settings.openai_api_key:
        return "openai"
    if shutil.which(settings.copilot_bin):
        return "copilot"
    raise RuntimeError("No LLM provider available: set OPENAI_API_KEY or install/login the GitHub Copilot CLI.")


def _install_cache(settings: Settings) -> None:
    global _cache_installed
    if settings.llm_cache and not _cache_installed:
        set_llm_cache(SQLiteLLMCache(settings.cache_dir / "llm_cache.sqlite"))
        _cache_installed = True


class UnavailableChatModel(BaseChatModel):
    """Placeholder used when no provider is configured: the app still starts and serves the
    deterministic features (lab flags, safety checks, knowledge graph); agent calls fail clearly."""

    reason: str = "No LLM provider configured."

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError(self.reason)


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()
    _install_cache(settings)
    try:
        provider = resolve_provider(settings)
    except RuntimeError as exc:
        return UnavailableChatModel(reason=str(exc))
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=temperature,
                          api_key=settings.openai_api_key, timeout=60, max_retries=settings.llm_retries)
    if provider == "copilot":
        return CopilotCLIChatModel(
            model=settings.copilot_model,
            copilot_bin=settings.copilot_bin,
            copilot_home=settings.copilot_home,
            timeout=settings.copilot_timeout,
            max_parallel=settings.copilot_max_parallel,
            retries=0,  # the CLI already retries internally; never multiply requests
            temperature=temperature,
        )
    raise RuntimeError("LLM_PROVIDER=fake must be injected explicitly (tests only).")


def describe_llm() -> dict:
    settings = get_settings()
    try:
        provider = resolve_provider(settings)
    except RuntimeError as exc:
        return {"provider": "unavailable", "error": str(exc)}
    model = settings.openai_model if provider == "openai" else settings.copilot_model
    return {"provider": provider, "model": model, "cache": settings.llm_cache}
