"""LangChain chat model backed by the local GitHub Copilot CLI.

Used during local development before an OpenAI key is available. Each call runs
`copilot -p <prompt> -s` in non-interactive mode with:
  * an isolated COPILOT_HOME (no MCP servers / plugins -> ~8s instead of ~70s per call)
  * no tools available (pure text completion)
  * the GitHub token from `gh auth token` (or GH_TOKEN / GITHUB_TOKEN)

Because it is a regular `BaseChatModel`, every LCEL chain (`prompt | llm | parser`)
works unchanged when switching to `ChatOpenAI`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_PREAMBLE = (
    "You are being used as a plain text-completion backend for a software application.\n"
    "Rules: do not use any tools, do not ask questions, do not add commentary.\n"
    "Respond ONLY with the content requested in the conversation below.\n"
)

# Linux MAX_ARG_STRLEN is 128 KiB for a single argv entry.
_MAX_PROMPT_BYTES = 120_000


class CopilotCLIError(RuntimeError):
    pass


_RATE_LIMIT_HINTS = ("rate limit", "rate-limit", "ratelimit", "429", "too many requests", "quota")


def is_rate_limited(message: str) -> bool:
    low = (message or "").lower()
    return any(h in low for h in _RATE_LIMIT_HINTS)


def resolve_github_token() -> str:
    for var in ("GH_TOKEN", "GITHUB_TOKEN", "COPILOT_GITHUB_TOKEN"):
        if os.getenv(var):
            return os.environ[var]
    if shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise CopilotCLIError("No GitHub token found. Run `gh auth login` or set GH_TOKEN.")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class CopilotCLIChatModel(BaseChatModel):
    """Chat model that delegates completions to `copilot -p`."""

    model: str = "gpt-4.1"
    copilot_bin: str = "copilot"
    copilot_home: Path = Field(default_factory=lambda: Path.home() / ".clinicalchat-copilot-home")
    timeout: int = 180
    max_parallel: int = 1
    retries: int = 0  # extra attempts after a transient (non rate-limit) failure
    temperature: float = 0.0  # accepted for API parity; the CLI does not expose it

    _semaphores: ClassVar[dict[int, threading.Semaphore]] = {}
    _token: ClassVar[Optional[str]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @property
    def _llm_type(self) -> str:
        return "copilot-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "provider": "copilot-cli"}

    def _semaphore(self) -> threading.Semaphore:
        with self._lock:
            if self.max_parallel not in self._semaphores:
                self._semaphores[self.max_parallel] = threading.Semaphore(self.max_parallel)
            return self._semaphores[self.max_parallel]

    def _get_token(self) -> str:
        with self._lock:
            if CopilotCLIChatModel._token is None:
                CopilotCLIChatModel._token = resolve_github_token()
            return CopilotCLIChatModel._token

    @staticmethod
    def render_messages(messages: List[BaseMessage]) -> str:
        parts = [_PREAMBLE]
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "SYSTEM INSTRUCTIONS"
            elif isinstance(msg, HumanMessage):
                role = "USER"
            elif isinstance(msg, AIMessage):
                role = "ASSISTANT"
            else:
                role = msg.type.upper()
            parts.append(f"=== {role} ===\n{msg.content}")
        return "\n\n".join(parts)

    def _run_cli(self, prompt: str) -> str:
        if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise CopilotCLIError("Prompt too large for the Copilot CLI backend; reduce retrieved context.")
        if not shutil.which(self.copilot_bin):
            raise CopilotCLIError(f"Copilot CLI binary '{self.copilot_bin}' not found on PATH.")

        self.copilot_home.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "COPILOT_HOME": str(self.copilot_home), "GH_TOKEN": self._get_token()}
        cmd = [
            self.copilot_bin, "-p", prompt, "-s",
            "--model", self.model,
            "--no-custom-instructions", "--disable-builtin-mcps",
            "--no-auto-update", "--no-remote", "--no-color",
            "--available-tools=",
        ]
        with self._semaphore(), tempfile.TemporaryDirectory(prefix="clinicalchat-llm-") as cwd:
            self._check_cooldown()
            started = time.time()
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # The CLI silently retries HTTP 429s for minutes; watch its log and stop it at the first 429
            # so a rate limit never turns into a burst of extra requests.
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if self._log_shows_rate_limit(proc.pid, started):
                        proc.kill()
                        proc.communicate()
                        self._trip_cooldown()
                        raise CopilotCLIError("429 rate limit reported by the Copilot backend")
                    if time.time() - started > self.timeout:
                        proc.kill()
                        proc.communicate()
                        raise CopilotCLIError(f"Copilot CLI timed out after {self.timeout}s")
        output = (stdout or "").strip()
        if proc.returncode != 0 or not output or output.startswith(("Error:", "Execution failed")):
            detail = (stderr or stdout or "").strip()[:500]
            if is_rate_limited(detail):
                self._trip_cooldown()
            raise CopilotCLIError(f"Copilot CLI failed (exit {proc.returncode}): {detail}")
        self._prune_state()
        return output

    # ---- rate-limit circuit breaker (shared by all instances in the process)
    _cooldown_until: ClassVar[float] = 0.0
    cooldown_seconds: int = 120

    def _check_cooldown(self) -> None:
        remaining = CopilotCLIChatModel._cooldown_until - time.time()
        if remaining > 0:
            raise CopilotCLIError(f"429 rate limit cool-down active ({int(remaining)}s left)")

    def _trip_cooldown(self) -> None:
        CopilotCLIChatModel._cooldown_until = time.time() + self.cooldown_seconds

    def _log_shows_rate_limit(self, pid: int, started: float) -> bool:
        logs = self.copilot_home / "logs"
        if not logs.exists():
            return False
        candidates = list(logs.glob(f"*-{pid}.log")) or [f for f in logs.glob("process-*.log") if f.stat().st_mtime >= started]
        for f in candidates:
            try:
                if '"status": 429' in f.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except OSError:
                pass
        return False

    def _prune_state(self, max_age_s: int = 3600) -> None:
        """The CLI writes a session folder per call; keep the isolated home small."""
        cutoff = time.time() - max_age_s
        for sub in ("session-state", "logs"):
            folder = self.copilot_home / sub
            if not folder.exists():
                continue
            for entry in folder.iterdir():
                try:
                    if entry.stat().st_mtime < cutoff:
                        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                except OSError:
                    pass

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self.render_messages(messages)
        started = time.time()
        attempt = 0
        while True:
            try:
                text = self._run_cli(prompt)
                break
            except CopilotCLIError as exc:
                if is_rate_limited(str(exc)):
                    # never retry into a rate limit: the breaker blocks further calls for a cool-down period
                    raise CopilotCLIError("The AI model is rate-limited right now (HTTP 429). Please wait a couple of "
                                          "minutes and ask again — or configure an OpenAI key.") from exc
                if attempt >= self.retries:
                    raise
                time.sleep(5 * 2 ** attempt)
                attempt += 1
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token, 1)[0]
        usage = {
            "input_tokens": _approx_tokens(prompt),
            "output_tokens": _approx_tokens(text),
            "total_tokens": _approx_tokens(prompt) + _approx_tokens(text),
        }
        message = AIMessage(
            content=text,
            usage_metadata=usage,
            response_metadata={"model_name": self.model, "provider": "copilot-cli",
                               "latency_s": round(time.time() - started, 2), "token_usage_estimated": True},
        )
        return ChatResult(generations=[ChatGeneration(message=message)],
                          llm_output={"model_name": self.model, "token_usage": usage})
