"""Knowledge engine: loads the corpus once and builds compact, patient-scoped context for each agent (no LLM)."""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Sequence

from clinicalchat.config import Settings, get_settings
from clinicalchat.ingestion.pipeline import load_corpus
from clinicalchat.ingestion.registry import PatientRegistry
from clinicalchat.knowledge_graph import MedicalKnowledgeGraph
from clinicalchat.models import ClinicalDocument, Domain, PatientProfile, SafetyAlert, Section
from clinicalchat.vectorstore import ClinicalVectorStore

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 9000

# panel keyword -> words that identify the matching lab section title
LAB_PANELS = {
    r"\bcbc\b|blood count|haemoglobin|hemoglobin|\bwbc\b|platelet|anaemi|anemi": ["cbc"],
    r"thyroid|\btsh\b|\bt3\b|\bt4\b": ["thyroid"],
    r"liver|\blft\b|bilirubin|\balt\b|\bast\b|sgpt|sgot": ["liver"],
    r"kidney|renal|\bkft\b|\brft\b|creatinine|urea|\begfr\b": ["kidney"],
    r"sugar|glucose|diabet|hba1c": ["sugar"],
    r"heart|cardiac|lipid|cholesterol|triglycer|troponin": ["heart"],
    r"\bbmi\b|body mass|weight": ["bmi"],
    r"metabolic|\bcmp\b|\bbmp\b|electrolyte": ["liver", "kidney", "sugar"],
}

IMAGING_TERMS = {
    "ct": r"\bct\b|computed tomograph", "mri": r"\bmri\b|magnetic resonance", "x-ray": r"x-?ray|\bcxr\b|radiograph",
    "ultrasound": r"ultrasound|\busg\b|sonograph|doppler", "chest": r"chest|lung|thora", "brain": r"brain|head|cranial",
    "abdomen": r"abdom|liver|appendi|gall|pelvi", "spine": r"spine|spinal|lumbar|cervical|disc",
    "mammogram": r"mammo|breast",
}


class ClinicalEngine:
    def __init__(self, settings: Optional[Settings] = None, documents: Optional[List[ClinicalDocument]] = None,
                 registry: Optional[PatientRegistry] = None, persist: bool = True):
        self.settings = settings or get_settings()
        if documents is None or registry is None:
            documents, registry = load_corpus(self.settings)
        self.documents = documents
        self.registry = registry
        self.kg = MedicalKnowledgeGraph(registry.all())
        self.store = ClinicalVectorStore(self.settings, persist=persist)
        self.store.index(documents)
        self.docs_by_id = {d.doc_id: d for d in documents}

    # ------------------------------------------------------------------ patients
    def patient(self, key: Optional[str]) -> Optional[PatientProfile]:
        return self.registry.get(key) if key else None

    def patient_choices(self) -> List[tuple]:
        return [(p.label, p.patient_key) for p in self.registry.all()]

    def find_patient_in_text(self, text: str) -> Optional[str]:
        """Patient named in the question (full name); prefers the record with the most data when names repeat."""
        matches = self.registry.search(text)
        if not matches:
            return None
        age = re.search(r"\b(\d{1,3})\s*(?:y|yr|yrs|years?)\b", text.lower())
        if age:
            exact = [p for p in matches if p.age == int(age.group(1))]
            matches = exact or matches
        return max(matches, key=lambda p: (len(p.name), len(p.section_ids))).patient_key

    def patient_domains(self, key: str) -> List[Domain]:
        return sorted({self.store.domain_of(s) for s in self.store.patient_sections(key)} - {None}, key=lambda d: d.value)

    def patient_summary(self, key: Optional[str]) -> str:
        p = self.patient(key)
        if not p:
            return "No specific patient selected."
        lines = [f"Name: {p.name}", f"Age: {p.age if p.age is not None else 'unknown'}", f"Gender: {p.gender or 'unknown'}"]
        lines.append("Documented allergies: " + (", ".join(f"{a.allergen} ({a.severity})" for a in p.allergies) or "none recorded"))
        if p.medications:
            lines.append("Medications: " + ", ".join(p.medications))
        conds = [c["condition"] for c in self.kg.patient_conditions(key)]
        if conds or p.conditions:
            lines.append("Conditions: " + ", ".join(dict.fromkeys([*p.conditions, *conds])))
        lines.append("Records: " + ", ".join(p.sources))
        return "\n".join(lines)

    # ------------------------------------------------------------------ section helpers
    def _source(self, s: Section) -> str:
        return self.store.source_of(s)

    @staticmethod
    def _fmt_section(s: Section, source: str, with_labs: bool = False, limit: int = 2500) -> str:
        head = f"### {s.title} — {s.patient_name or 'reference'} [{source}]" + (f" ({s.report_date})" if s.report_date else "")
        if with_labs and s.lab_results:
            body = "\n".join("- " + r.line() for r in s.lab_results)
        else:
            body = s.text[:limit]
        return f"{head}\n{body}"

    def _pack(self, chunks: Sequence[str]) -> str:
        out, used = [], 0
        for c in chunks:
            if used + len(c) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - used
                if remaining > 400:
                    out.append(c[:remaining] + "…")
                break
            out.append(c)
            used += len(c)
        return "\n\n".join(out) if out else "No records retrieved."

    def _reference(self, query: str, domains: Sequence[Domain], k: int = 3) -> List[str]:
        hits = self.store.search(query, k=k, domains=[*domains, Domain.REFERENCE], patient_key="", include_general=False)
        return [f"### Reference: {h.metadata.get('title', '')} [{h.source}]\n{h.text}" for h in hits if h.score > 0.15]

    # ------------------------------------------------------------------ agent contexts
    def lab_context(self, query: str, key: Optional[str]) -> tuple[str, List[str]]:
        q = query.lower()
        if key:
            sections = [s for s in self.store.patient_sections(key, [Domain.LAB]) if s.lab_results or s.text.strip()]
            wanted = {w for rx, words in LAB_PANELS.items() if re.search(rx, q) for w in words}
            if wanted:
                focused = [s for s in sections if any(w in (s.title + " " + self._source(s)).lower() for w in wanted)]
                sections = focused or sections
            chunks = [self._fmt_section(s, self._source(s), with_labs=True) for s in sections]
        else:
            hits = self.store.search(query, k=6, domains=[Domain.LAB])
            sections = self.store.sections_for(hits, max_sections=3)
            chunks = [self._fmt_section(s, self._source(s), with_labs=True) for s in sections]
        chunks += self._reference(query, [Domain.LAB], k=2)
        sources = list(dict.fromkeys(self._source(s) for s in sections))
        return self._pack(chunks), sources

    def radiology_context(self, query: str, key: Optional[str]) -> tuple[str, List[str]]:
        q = query.lower()
        wanted = [term for term, rx in IMAGING_TERMS.items() if re.search(rx, q)]
        if key:
            sections = self.store.patient_sections(key, [Domain.RADIOLOGY])
        else:
            hits = self.store.search(query, k=8, domains=[Domain.RADIOLOGY])
            sections = self.store.sections_for(hits, max_sections=4)
        if wanted:
            def score(s: Section) -> int:
                text = (s.title + " " + s.text[:300]).lower()
                return sum(bool(re.search(IMAGING_TERMS[t], text)) for t in wanted)
            best = max((score(s) for s in sections), default=0)
            if best:
                sections = [s for s in sections if score(s) == best]
        chunks = [self._fmt_section(s, self._source(s)) for s in sections]
        chunks += self._reference(query, [Domain.RADIOLOGY], k=1)
        sources = list(dict.fromkeys(self._source(s) for s in sections))
        return self._pack(chunks), sources

    def allergy_context(self, query: str, key: Optional[str], drugs: Sequence[str], allergens: Sequence[str]) -> tuple[str, List[str]]:
        p = self.patient(key)
        chunks, sources = [], []
        if p and p.allergies:
            for a in p.allergies:
                chunks.append(f"### Documented allergy: {a.allergen} [{', '.join(a.evidence[:1]) or a.source}]\n"
                              f"Category: {a.category}; Severity: {a.severity}; Reactions: {', '.join(a.reactions) or 'n/a'}; "
                              f"Emergency medication: {a.emergency_medication or 'n/a'}; Restrictions: {a.restrictions or 'n/a'}; "
                              f"Alternatives: {a.alternatives or 'n/a'}")
                sources.append(a.source)
            for s in self.store.patient_sections(key, [Domain.ALLERGY]):
                chunks.append(self._fmt_section(s, self._source(s), limit=1500))
                sources.append(self._source(s))
        elif p and p.negative_allergy_tests:
            chunks.append("Negative allergy tests: " + ", ".join(p.negative_allergy_tests))
        hits = self.store.search(query, k=3, domains=[Domain.ALLERGY, Domain.REFERENCE],
                                 patient_key="" if key else None, include_general=False)
        chunks += [f"### Reference: {h.metadata.get('title', '')} [{h.source}]\n{h.text}" for h in hits if h.score > 0.15]
        sources += [h.source for h in hits if h.score > 0.15]
        return self._pack(chunks), [s for s in dict.fromkeys(sources) if s]

    def coordinator_context(self, query: str, key: Optional[str]) -> tuple[str, List[str]]:
        p = self.patient(key)
        if not p:
            hits = self.store.search(query, k=6)
            sections = self.store.sections_for(hits, max_sections=4)
            return self._pack([self._fmt_section(s, self._source(s), limit=1200) for s in sections]), \
                list(dict.fromkeys(self._source(s) for s in sections))
        sections = self.store.patient_sections(key)
        chunks = []
        lab_lines = []
        for s in sections:
            if not s.lab_results:
                continue
            abnormal = [r for r in s.lab_results if r.abnormal]
            status = "; ".join(r.line() for r in abnormal) if abnormal else f"all {len(s.lab_results)} values within reference range"
            lab_lines.append(f"- {s.title} ({s.report_date or 'undated'}) [{self._source(s)}]: {status}")
        if lab_lines:
            chunks.append("### Lab panels\n" + "\n".join(lab_lines))
        for s in sections:
            dom = self.store.domain_of(s)
            if dom == Domain.LAB:
                continue  # summarised above
            chunks.append(self._fmt_section(s, self._source(s), limit=1200))
        review = self.kg.medication_review(key)
        if review:
            chunks.append("### Medication safety review (knowledge graph)\n" + "\n".join(f"- {alert_line(a)}" for a in review))
        return self._pack(chunks), list(dict.fromkeys(self._source(s) for s in sections))

    # ------------------------------------------------------------------ deterministic safety (no LLM)
    def safety_alerts(self, key: Optional[str], drugs: Sequence[str]) -> List[SafetyAlert]:
        alerts: List[SafetyAlert] = []
        for drug in drugs:
            alerts += self.kg.check_drug_for_patient(drug, key) if key else self.kg.check_drug(drug)
        uniq = {(a.level, a.category, a.drug, a.against, a.message): a for a in alerts}
        return sorted(uniq.values(), key=lambda a: a.rank)

    def allergy_facts(self, key: Optional[str], drugs: Sequence[str], allergens: Sequence[str]) -> str:
        lines = []
        for a in self.safety_alerts(key, drugs):
            lines.append("ALERT " + alert_line(a))
        p = self.patient(key)
        names = [*allergens, *[x.allergen for x in (p.allergies if p else [])]]
        seen = set()
        for name in names:
            cross = self.kg.cross_reactive(name)
            sig = tuple(c["allergen"] for c in cross) or (name.lower(),)
            if sig in seen:
                continue
            seen.add(sig)
            if cross:
                lines.append(f"Cross-reactivity for {name}: " + "; ".join(f"{c['allergen']} ({c['risk']} risk: {c['reason']})" for c in cross))
            alts = self.kg.safe_alternatives(name)
            if alts:
                lines.append(f"Safe alternatives for {name}: " + ", ".join(alts))
        if drugs and not any(l.startswith("ALERT") for l in lines):
            lines.append(f"No allergy/interaction conflict found in the knowledge graph for: {', '.join(drugs)}"
                         + ("" if p else " (no patient selected — patient allergies not checked)"))
        return "\n".join(lines) or "None."


def alert_line(a: SafetyAlert) -> str:
    tail = f" — {a.management}" if a.management else ""
    return f"[{a.level}] {a.drug} vs {a.against}: {a.message}{tail}"


_ENGINE: Optional[ClinicalEngine] = None
_LOCK = threading.Lock()


def get_engine() -> ClinicalEngine:
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = ClinicalEngine()
        return _ENGINE
