"""Small persistent LangChain LLM cache (SQLite) to avoid re-paying for identical prompts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation


class SQLiteLLMCache(BaseCache):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.commit()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\x00{prompt}".encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM llm_cache WHERE key=?", (self._key(prompt, llm_string),)).fetchone()
        if not row:
            return None
        try:
            items = json.loads(row[0])
        except ValueError:
            return None
        generations: list[Generation] = []
        for item in items:
            if item.get("kind") == "chat":
                msg = AIMessage(content=item["text"], usage_metadata=item.get("usage"),
                                response_metadata={**item.get("meta", {}), "cache_hit": True})
                generations.append(ChatGeneration(message=msg))
            else:
                generations.append(Generation(text=item["text"]))
        return generations

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        items = []
        for gen in return_val:
            if isinstance(gen, ChatGeneration):
                msg = gen.message
                items.append({"kind": "chat", "text": msg.content if isinstance(msg.content, str) else str(msg.content),
                              "usage": getattr(msg, "usage_metadata", None),
                              "meta": {k: v for k, v in (msg.response_metadata or {}).items()
                                       if isinstance(v, (str, int, float, bool))}})
            else:
                items.append({"kind": "text", "text": gen.text})
        payload = json.dumps(items)
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
                               (self._key(prompt, llm_string), payload))
            self._conn.commit()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

