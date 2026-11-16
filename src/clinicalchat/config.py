"""Configuration from environment variables / `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(os.getenv("CLINICALCHAT_HOME") or Path(__file__).resolve().parents[2]).resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    work_dir: Path = field(default_factory=lambda: _path("WORK_DIR", PROJECT_ROOT / "data"))

    # LLM: "auto" = OpenAI when OPENAI_API_KEY is set, otherwise the local GitHub Copilot CLI
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # strong model for the clinician-facing Insight agent, fast model for patient-facing answers
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1"))
    openai_fast_model: str = field(default_factory=lambda: os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-mini"))
    openai_timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")))
    # output length is what drives LLM latency: hard caps per agent
    insight_max_tokens: int = field(default_factory=lambda: int(os.getenv("INSIGHT_MAX_TOKENS", "900")))
    communication_max_tokens: int = field(default_factory=lambda: int(os.getenv("COMMUNICATION_MAX_TOKENS", "450")))
    copilot_model: str = field(default_factory=lambda: os.getenv("COPILOT_MODEL", "gpt-4.1"))
    copilot_bin: str = field(default_factory=lambda: os.getenv("COPILOT_BIN", "copilot"))
    # one call at a time keeps the Copilot backend well under its rate limits
    copilot_max_parallel: int = field(default_factory=lambda: int(os.getenv("COPILOT_MAX_PARALLEL", "1")))
    copilot_timeout: int = field(default_factory=lambda: int(os.getenv("COPILOT_TIMEOUT_SECONDS", "180")))
    llm_cache: bool = field(default_factory=lambda: _bool("LLM_CACHE", True))
    llm_retries: int = field(default_factory=lambda: int(os.getenv("LLM_RETRIES", "2")))

    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "auto").lower())
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    ocr_enabled: bool = field(default_factory=lambda: _bool("OCR_ENABLED", True))

    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "7860")))
    demo_password: str = field(default_factory=lambda: os.getenv("DEMO_PASSWORD", "demo@123"))

    @property
    def processed_dir(self) -> Path:
        return self.work_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.work_dir / "chroma"

    @property
    def audit_dir(self) -> Path:
        return self.work_dir / "audit"

    @property
    def export_dir(self) -> Path:
        return self.work_dir / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def copilot_home(self) -> Path:
        return _path("COPILOT_HOME_DIR", self.project_root / ".copilot-llm-home")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.chroma_dir, self.audit_dir, self.export_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
