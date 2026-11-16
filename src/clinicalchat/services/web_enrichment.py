"""🌐 Web enrichment (Tavily, trusted sites only): general health answers for patients, guideline evidence for staff,
and look-ups of medicines the knowledge graph does not know. Queries are de-identified; the web only *adds*
information and checks — it never clears a safety warning or overrides the patient's records."""

from __future__ import annotations

import re
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Iterator, List, Optional, Sequence

from clinicalchat.domain.conversation import Event
from clinicalchat.domain.models import Domain
from clinicalchat.domain.roles import policy
from clinicalchat.domain.web_evidence import WebResult, WebSearchOutcome
from clinicalchat.infrastructure.web_search import TavilySearch
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.router import RoutePlan
from clinicalchat.services.turn import TurnContext

WEB_WAIT_PATIENT = 2.5  # seconds a patient's general answer waits for web sources (then answers without them)
SITES = {"health": "WHO, NIH, CDC, NHS, Mayo Clinic…", "guideline": "NICE, NCBI/PubMed, KDIGO, AHA…",
         "drug": "DailyMed, MedlinePlus, drugs.com…"}
GENERIC_WORDS = set("""what which when where does should would could their there this that these those with from about
have patient patients condition conditions problem problems issue issues findings diagnosis current currently best
options option next steps step approach plan care give tell explain summary summarize please need needs into
right based results result report reports labs case history them they""".split())
GUIDELINE_ASK = re.compile(
    r"\b(guidelines?|guidance|evidence|literature|research|stud(?:y|ies)|trials?|latest|recommend\w*|protocols?|"
    r"first[- ]line|treat\w*|manag\w*|therap\w*|dos(?:e|es|ing)|standard of care|criteria|work[- ]?up|screening)\b", re.I)
DRUG_CANDIDATE = re.compile(
    r"\b(?:take|taking|took|use|using|give|giving|prescribed|start(?:ed|ing)?|stop(?:ped|ping)?|tablets? of|syrup)\s+"
    r"(?:taking |using )?(?:a |an |the |some |one |two |this |my |his |her )?"
    r"([A-Za-z][A-Za-z0-9-]{2,}(?:[ -](?:\d{2,4}|forte|plus|ds|xr|sr|er|cr|duo|max|mr))?)", re.I)
DRUG_SAFE_ASK = re.compile(r"\b(?:is|are)\s+([A-Za-z][A-Za-z0-9-]{2,}(?:[ -]\d{2,4})?)\s+(?:safe|ok|okay|fine|allowed)\b", re.I)
NOT_DRUGS = set("""medicine medicines medication medications tablet tablets pill pills drug drugs capsule capsules it this that
them these those rest care food foods water alcohol exercise bath shower leave test tests report reports painkiller
painkillers antibiotic antibiotics something anything vaccine vaccines steps action time walk coffee tea milk sugar
juice supplement supplements vitamin vitamins iron calcium injection injections blood more less any dose doses course
diet sleep nap break off over care advice help them treatment surgery holiday day days week weeks it's while place
fever cold cough pain""".split())


def wants_guidelines(question: str) -> bool:
    return bool(GUIDELINE_ASK.search(question))


def guideline_query(question: str, conditions: Sequence[str]) -> str:
    """"guideline for her condition" names no condition -> add the top findings; "…for hyperkalemia" stays as is."""
    vague = not (set(re.findall(r"[a-z][a-z-]{3,}", GUIDELINE_ASK.sub(" ", question.lower()))) - GENERIC_WORDS)
    terms = " ".join(conditions[:2]) if vague else ""
    return f"{question} {terms} clinical guideline"


class WebEnrichment:
    def __init__(self, web: TavilySearch, patients: PatientDirectory, kg: MedicalKnowledgeGraph):
        self.web = web
        self.patients = patients
        self.kg = kg
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web")

    @property
    def enabled(self) -> bool:
        return self.web.enabled

    def start(self, kind: str, query: str, ctx: TurnContext) -> Optional[Future]:
        """Start a search in the background (collect it later with ``search(..., started=...)``)."""
        if not self.web.enabled:
            return None
        return self._pool.submit(self.web.search, query, kind, self.patients.names(), ctx.llm_config(ctx.trace))

    def search(self, kind: str, query: str, ctx: TurnContext, wait: Optional[float] = None,
               started: Optional[Future] = None) -> Iterator[Event]:
        """Search trusted sites (or collect a search ``started`` earlier); returns the outcome, or None when off / too
        slow for ``wait`` seconds."""
        fut = started or self.start(kind, query, ctx)
        if fut is None:
            return None
        yield Event("status", "🌐 Web search — trusted sites", f"Searching {SITES[kind]} (no personal details are sent)",
                    "pending", "web")
        try:
            out = fut.result(timeout=wait)
        except FutureTimeout:  # answer now; the search finishes in the background and is cached for next time
            yield Event("status", "🌐 Web search — skipped", f"Took longer than {wait:.0f}s — answering without it",
                        "done", "web")
            return None
        ctx.run.web.append(out)
        yield Event("status", "🌐 Web search — done", _summary(out), "done", "web")
        return out

    def lookup_unknown_drugs(self, ctx: TurnContext) -> Iterator[Event]:
        """A medicine the knowledge graph does not know (often a brand, e.g. "Dolo 650", "Augmentin"): look up what it
        contains on trusted drug sites, then let the normal deterministic allergy / interaction checks run on those
        ingredients."""
        if not self.web.enabled:
            return
        for cand in self._drug_candidates(ctx.question, ctx.plan):
            out = yield from self.search("drug", f"{cand} medicine active ingredient generic name", ctx)
            if out is not None:
                self._apply_drug_lookup(cand, out, ctx)

    def _drug_candidates(self, question: str, plan: RoutePlan) -> List[str]:
        known = {d.lower() for d in plan.drugs}
        found = []
        for m in [*DRUG_CANDIDATE.finditer(question), *DRUG_SAFE_ASK.finditer(question)]:
            cand = m.group(1).strip(" -")
            head = cand.split()[0].lower()
            if (head in NOT_DRUGS or head in known or self.kg.resolve_drugs(cand) or self.kg.resolve_allergen(cand)
                    or self.kg.symptoms_in_text(cand)):
                continue
            found.append(cand)
        return list(dict.fromkeys(found))[:2]

    def _apply_drug_lookup(self, cand: str, out: WebSearchOutcome, ctx: TurnContext) -> None:
        plan = ctx.plan
        head = cand.split()[0].lower()
        matching: List[WebResult] = [r for r in out.results if head in f"{r.title} {r.content}".lower()][:3]
        ingredients = list(dict.fromkeys(d for r in matching for d in self.kg.resolve_drugs(f"{r.title} {r.content}")))[:3]
        if ingredients:
            src = matching[0]
            plan.drugs = list(dict.fromkeys([*plan.drugs, *ingredients]))
            plan.check_allergies = True
            if Domain.ALLERGY in policy(ctx.session.role).domains and Domain.ALLERGY not in plan.focus:
                plan.focus.append(Domain.ALLERGY)
            plan.drug_notes.append(f"🌐 **{cand}** — according to [{src.domain}]({src.url}) it contains "
                                   f"**{', '.join(ingredients)}**; checked below. _Web lookup — confirm with a pharmacist._")
            plan.notes.append(f"Web lookup: {cand} → {', '.join(ingredients)}")
        elif matching:
            plan.drug_notes.append(f"🌐 **{cand}** — found on [{matching[0].domain}]({matching[0].url}), but its "
                                   "ingredients are not in our safety database, so it could **not** be checked against "
                                   "allergies. Please ask a pharmacist or your doctor before taking it.")


def _summary(out: WebSearchOutcome) -> str:
    if out.error:
        return f"Unavailable ({out.error[:80]})"
    sites = ", ".join(dict.fromkeys(r.domain for r in out.results)) or "nothing relevant"
    return f"{len(out.results)} result(s): {sites}" + (" (cached)" if out.cached else f" ({out.latency_ms} ms)")
