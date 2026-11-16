"""Tavily web search, restricted to trusted medical sites.

* ``health``    — patient general questions (WHO, CDC, NIH/MedlinePlus, NHS, Mayo Clinic…)
* ``guideline`` — clinician guideline / treatment evidence (NICE, NCBI/PubMed, KDIGO, AHA…)
* ``drug``      — what an unknown medicine / brand contains (DailyMed, MedlinePlus, drugs.com…)

Queries pass through :func:`deidentify` before they leave the server; results are cached for a week.
Web results never change risk levels or allergy/drug decisions — those stay deterministic.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from typing import Iterable, List, Optional

import httpx

from clinicalchat.config import Settings
from clinicalchat.domain.privacy import deidentify
from clinicalchat.domain.web_evidence import WebResult, WebSearchOutcome
from clinicalchat.infrastructure.repositories.caches import WebCache

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


def trusted(domain: str, kind: str) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in DOMAINS[kind])


class TavilySearch:
    def __init__(self, settings: Settings, cache: Optional[WebCache] = None):
        self.settings = settings
        self.cache = cache
        self._http = httpx.Client(timeout=settings.web_search_timeout, http2=False)

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
        cached = self.cache.lookup(kind, clean) if self.cache else None
        if cached is not None:
            out.results, out.cached = [WebResult(**r) for r in cached], True
        else:
            try:
                out.results = self._traced(clean, kind, config)
                if self.cache:
                    self.cache.save(kind, clean, [asdict(r) for r in out.results])
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
            if (res.url.startswith("https://") and trusted(res.domain, kind) and res.content
                    and not seen & {res.url, res.title.lower()}):
                seen |= {res.url, res.title.lower()}
                results.append(res)
        return results
