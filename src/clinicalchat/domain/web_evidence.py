"""Results of a trusted-site web search, and how they are shown to the AI ([W1]…) and to people (links)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse


@dataclass
class WebResult:
    title: str
    url: str
    content: str
    score: float = 0.0

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")


@dataclass
class WebSearchOutcome:
    kind: str
    query: str
    results: List[WebResult] = field(default_factory=list)
    cached: bool = False
    error: str = ""
    latency_ms: int = 0

    def audit(self) -> dict:
        return {"kind": self.kind, "query": self.query, "results": len(self.results), "cached": self.cached,
                "error": self.error, "latency_ms": self.latency_ms, "domains": [r.domain for r in self.results]}

    def context(self, limit: int = 3, chars: int = 700) -> str:
        """Numbered evidence block for a prompt ([W1], [W2]…)."""
        return "\n\n".join(f"[W{i}] {r.title} ({r.domain})\n{r.content[:chars]}"
                           for i, r in enumerate(self.results[:limit], 1))

    def links_markdown(self, title: str, limit: int = 3) -> str:
        if not self.results:
            return ""
        items = "\n".join(f"- [W{i}] [{_md(r.title)}]({r.url}) — {r.domain}" for i, r in enumerate(self.results[:limit], 1))
        return f"\n\n**{title}**\n{items}"


def _md(text: str) -> str:
    return re.sub(r"[\[\]]", "", text).strip()[:110]
