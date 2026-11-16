"""Tavily web search, restricted to trusted medical sites.

Three uses:
* ``health``    — patient general questions (WHO, CDC, NIH/MedlinePlus, NHS, Mayo Clinic…)
* ``guideline`` — clinician guideline / treatment evidence (NICE, NCBI/PubMed, KDIGO, AHA…)
* ``drug``      — what an unknown medicine / brand contains (DailyMed, MedlinePlus, drugs.com…)

Privacy: queries are built from medical terms only and pass through :func:`deidentify` (patient names, IDs,
dates, phone numbers and e-mails are removed) before they leave the server. Results are cached in SQLite.
Web results never change risk levels or allergy/drug decisions — those stay deterministic.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import httpx

from clinicalchat.config import Settings, get_settings

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

DOMAINS = {
    "health": ["medlineplus.gov", "nih.gov", "cdc.gov", "who.int", "nhs.uk", "mayoclinic.org", "clevelandclinic.org",
               "heart.org", "diabetes.org", "kidney.org"],
    "guideline": ["nice.org.uk", "ncbi.nlm.nih.gov", "who.int", "cdc.gov", "nih.gov", "kdigo.org", "heart.org",
                  "acc.org", "idsociety.org", "diabetesjournals.org", "aafp.org", "thoracic.org", "gastro.org",
                  "acr.org", "radiopaedia.org"],
    "drug": ["dailymed.nlm.nih.gov", "medlineplus.gov", "drugs.com", "nhs.uk", "rxlist.com", "1mg.com",
             "ema.europa.eu", "fda.gov"],
}

_SCRUB = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                                  # e-mail
    re.compile(r"(?i)\b(patient\s+name|name)\s*[:=]\s*\S+(\s+\S+)?"),           # "Name: Robin Hood"
    re.compile(r"(?i)\b(mrn|dob|uhid|id|case|patient\s+id)\b\s*[:=#]?\s*[\w/.-]*\d[\w/.-]*"),
    re.compile(r"\+?\d[\d\s()-]{7,}\d"),                                     # phone numbers
    re.compile(r"\b\d{6,}\b"),                                               # long ids
    re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b"),                      # dates
    re.compile(r"\b[A-Z]{1,4}-[A-Z0-9]{4,}\b"),                               # case ids (C-5A0A12FB)
    re.compile(r"\b[a-z]+(?:-[a-z]+)+-\d{1,3}(?:-\d+)?\b"),                  # patient keys (robin-hood-31)
]


def deidentify(text: str, names: Iterable[str] = ()) -> str:
    """Remove anything that could identify a patient before a query leaves the server."""
    for rx in _SCRUB:
        text = rx.sub(" ", text)
    parts = {p for n in names for p in re.split(r"[\s.'-]+", (n or "").lower()) if len(p) >= 3}
    if parts:
        text = re.sub(r"(?i)\b(" + "|".join(map(re.escape, sorted(parts, key=len, reverse=True))) + r")(?:'s)?\b",
                      " ", text)
    text = re.sub(r"\(\s*[,;]?\s*\)|\s+[,;](?=[\s,;)])", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;:-()")[:380]


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


class WebSearch:
    def __init__(self, settings: Optional[Settings] = None, store=None):
        self.settings = settings or get_settings()
        self.store = store
        self._http = httpx.Client(timeout=self.settings.web_search_timeout, http2=False)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.tavily_api_key) and self.settings.web_search != "off"

    def search(self, query: str, kind: str, names: Iterable[str] = (), config: Optional[dict] = None) -> WebSearchOutcome:
        clean = deidentify(query, names)
        out = WebSearchOutcome(kind=kind, query=clean)
        if not self.enabled or len(clean) < 3:
            out.error = "" if self.enabled else "web search is off"
            return out
        start = time.perf_counter()
        cached = self.store.web_lookup(kind, clean) if self.store else None
        if cached is not None:
            out.results, out.cached = [WebResult(**r) for r in cached], True
        else:
            try:
                out.results = self._traced(clean, kind, config)
                if self.store:
                    self.store.web_save(kind, clean, [asdict(r) for r in out.results])
            except Exception as exc:  # the web is optional: never break the chat
                out.error = f"{type(exc).__name__}: {str(exc)[:160]}"
                log.warning("Web search failed (%s): %s", kind, out.error)
        out.latency_ms = int((time.perf_counter() - start) * 1000)
        return out

    def _traced(self, query: str, kind: str, config: Optional[dict]) -> List[WebResult]:
        """Run the search as a LangChain runnable so it appears as a span in Langfuse next to the agents."""
        from langchain_core.runnables import RunnableLambda

        cfg = {**(config or {}), "run_name": f"🌐 Web search ({kind})",
               "metadata": {**(config or {}).get("metadata", {}), "tool": "tavily", "kind": kind}}
        return RunnableLambda(lambda q: self._call(q, kind)).invoke(query, config=cfg)

    def _call(self, query: str, kind: str) -> List[WebResult]:
        resp = self._http.post(TAVILY_URL,
                          headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
                          json={"query": query, "search_depth": "basic", "topic": "general",
                                "max_results": self.settings.web_search_max_results,
                                "include_domains": DOMAINS[kind], "include_answer": False})
        resp.raise_for_status()
        results, seen = [], set()
        for r in resp.json().get("results", []):
            res = WebResult(title=str(r.get("title", "")).strip(), url=str(r.get("url", "")).split("#")[0],
                            content=re.sub(r"\s+", " ", str(r.get("content", ""))).strip(), score=float(r.get("score") or 0))
            key = (res.url.split("#")[0], res.title.lower())
            if res.url.startswith("https://") and trusted(res.domain, kind) and res.content and not seen & {key[0], key[1]}:
                seen |= {key[0], key[1]}
                results.append(res)
        return results


def trusted(domain: str, kind: str) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in DOMAINS[kind])
