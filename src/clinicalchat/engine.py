"""Knowledge engine: loads the corpus once and builds compact, patient-scoped context for each agent (no LLM)."""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, List, Optional, Sequence

from clinicalchat.config import Settings, get_settings
from clinicalchat.ingestion.pipeline import load_corpus
from clinicalchat.ingestion.registry import PatientRegistry, patient_key
from clinicalchat.knowledge_graph import MedicalKnowledgeGraph
from clinicalchat.models import AllergyRecord, ClinicalDocument, Domain, PatientProfile, SafetyAlert, Section
from clinicalchat.vectorstore import ClinicalVectorStore

if TYPE_CHECKING:
    from clinicalchat.store import Store

log = logging.getLogger(__name__)
SELF_REPORTED = "Self-reported by patient"

MAX_CONTEXT_CHARS = 9000

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
        self._lock = threading.RLock()

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
        if p.symptoms:
            lines.append("Reported symptoms: " + "; ".join(p.symptoms[-5:]))
        lines.append("Records: " + (", ".join(p.sources) or "none on file yet"))
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

    def records_context(self, query: str, sections: Sequence[Section], domains: Sequence[Domain], reference_k: int = 2) -> str:
        """Compact text of the selected records (+ a little reference material) for the Insight agent."""
        chunks = [self._fmt_section(s, self._source(s), with_labs=bool(s.lab_results), limit=1800) for s in sections]
        if reference_k:
            chunks += self._reference(query, [d for d in domains if d != Domain.REFERENCE], k=reference_k)
        return self._pack(chunks)

    def reference_context(self, query: str, k: int = 4) -> tuple[str, List[str]]:
        hits = self.store.search(query, k=k, patient_key="", include_general=False)
        hits = [h for h in hits if h.score > 0.1]
        return (self._pack([f"### {h.metadata.get('title', '')} [{h.source}]\n{h.text}" for h in hits]),
                list(dict.fromkeys(h.source for h in hits)))

    # ------------------------------------------------------------------ self-registered patients & uploads
    def new_patient_key(self, name: str, age: Optional[int]) -> str:
        """Never merge a self-registered user into an existing record with the same name (privacy)."""
        base = patient_key(name, age)
        key, n = base, 1
        while key in self.registry.patients:
            n += 1
            key = f"{base}-{n}"
        return key

    def create_patient(self, key: str, name: str, age: Optional[int], gender: Optional[str]) -> PatientProfile:
        with self._lock:
            profile = self.registry.patients.get(key) or PatientProfile(patient_key=key, name=name, age=age, gender=gender)
            self.registry.patients[key] = profile
            self.kg.refresh_patient(profile)
            return profile

    def apply_fact(self, key: str, kind: str, value: str, refresh: bool = True) -> None:
        p = self.patient(key)
        if not p or not value.strip():
            return
        with self._lock:
            if kind == "symptom" and value not in p.symptoms:
                p.symptoms.append(value)
            elif kind == "allergy" and value.lower() not in {a.allergen.lower() for a in p.allergies}:
                allergen = self.kg.resolve_allergen(value) or value.strip()
                p.allergies.append(AllergyRecord(allergen=allergen, category="self-reported", severity="UNKNOWN",
                                                 source=SELF_REPORTED, evidence=[SELF_REPORTED]))
            elif kind == "medication" and value not in p.medications:
                p.medications.append(value)
            if refresh:
                self.kg.refresh_patient(p)

    def add_document(self, doc: ClinicalDocument) -> List[str]:
        with self._lock:
            if doc.doc_id in self.docs_by_id:
                return []
            self.documents.append(doc)
            self.docs_by_id[doc.doc_id] = doc
            self.store.add_document(doc)
            touched = self.registry.add_document(doc)
            for key in touched:
                self.kg.refresh_patient(self.registry.patients[key])
            return touched

    def load_user_data(self, store: "Store") -> None:
        for u in store.users():
            if u["patient_key"] not in self.registry.patients:
                self.registry.patients[u["patient_key"]] = PatientProfile(
                    patient_key=u["patient_key"], name=u["display_name"], age=u["age"], gender=u["gender"])
        for f in store.facts():
            self.apply_fact(f["patient_key"], f["kind"], f["value"], refresh=False)
        for row in store.uploads():
            try:
                self.add_document(ClinicalDocument.model_validate_json(row["doc_json"]))
            except Exception:  # a corrupt upload must not stop the app
                log.exception("Could not reload upload %s", row["doc_id"])
        for u in store.users():
            self.kg.refresh_patient(self.registry.patients[u["patient_key"]])

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
            from clinicalchat.store import get_store
            engine = ClinicalEngine()
            engine.load_user_data(get_store())
            _ENGINE = engine
        return _ENGINE
