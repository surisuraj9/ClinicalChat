"""🔬 Clinical Analysis Agent (deterministic, no LLM).

Contains the three specialist analysers from the problem statement and the symptom reasoning from the slide:
* Lab Interpreter      — values vs printed reference ranges, abnormal patterns -> possible conditions (KG TEST_INDICATES)
* Radiology Analyzer   — findings / impression / red flags in imaging reports (negation-aware)
* Allergy Safety       — documented allergies, cross-reactivity, proposed-drug and current-medication checks (KG)
* Symptom reasoning    — symptom -> condition differential (KG) and "Clinical Inconsistency" when symptoms are not
                         explained by the reports or a report's own flag contradicts its reference range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from clinicalchat.extraction import ExtractionResult
from clinicalchat.kg_ontology import TEST_INDICATES
from clinicalchat.models import Domain, LabResult, SafetyAlert, Section

if TYPE_CHECKING:
    from clinicalchat.engine import ClinicalEngine

_IMAGING_RED_FLAGS = ["haemorrhage", "hemorrhage", "infarct", "pneumothorax", r"(?<!body )\bmass\b", "malignan", "metasta",
                      "fracture", "dissection", "stroke", "midline shift", "cord compression", "free air", "tumou?r",
                      "aneurysm", "carcinoma", "appendicitis", r"\bsuspicious\b", r"pi.?rads [45]"]
_ABNORMAL_IMAGING = re.compile(r"consolidation|effusion|pneumonia|pneumonitis|ground.?glass|cardiomegaly|nodule|lesion|"
                               r"opacit|infiltrat|edema|oedema|stenosis|herniat|collapse|atelecta|thicken|calcul|stone|"
                               r"cyst|abnormal|enlarg|degenerat|ischemi|ischaemi|hypodensit", re.I)
_NEGATION = re.compile(r"\b(no|without|negative for|free of|absence of|not seen|ruled out|rule out)\b[^.\n]{0,40}$", re.I)
_SKIP_HEADER = re.compile(r"normal values|normal range|threshold|classification|reference|why |analysis|guide|quantitative", re.I)
_SKIP_LINE = re.compile(r"→|->|\(normal|normal:|expected|\b0\s?(?:mm|cm)\b", re.I)
_INLINE_FIELD = re.compile(r"^[A-Za-z /&()\-]{2,40}:\s*\S")
_IMPRESSION = re.compile(r"(?:impression|conclusion)\s*:?\s*(.+)", re.I)


def _report_lines(text: str, title: str = ""):
    header = ""
    for line in text.splitlines():
        st = line.strip(" \t•-*")
        if not st or (title and st == title.strip()):
            continue
        if st.endswith(":") and len(st) < 60:
            header = st
            continue
        if _INLINE_FIELD.match(st):
            header = ""
        if _SKIP_HEADER.search(header) or _SKIP_LINE.search(st):
            continue
        yield header, st


def imaging_red_flags(text: str, title: str = "") -> List[str]:
    hits = []
    for _, st in _report_lines(text, title):
        low = st.lower()
        for flag in _IMAGING_RED_FLAGS:
            m = re.search(flag, low)
            if m and not _NEGATION.search(low[:m.start()][-60:]):
                hits.append(st)
                break
    return list(dict.fromkeys(hits))


def imaging_abnormal(text: str, title: str = "") -> List[str]:
    hits = []
    for header, st in _report_lines(text, title):
        if not re.search(r"finding|impression|diagnos|observation|conclusion|comparison", header + " " + st.split(":")[0], re.I):
            continue
        m = _ABNORMAL_IMAGING.search(st)
        if m and not _NEGATION.search(st[:m.start()][-60:]) and not re.search(r"\bno\b|\bnormal\b(?! limits)", st[:m.start()], re.I):
            hits.append(st)
    return list(dict.fromkeys(hits))


def impression_of(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _IMPRESSION.match(line.strip(" \t•-*"))
        if m:
            rest = m.group(1).strip()
            if not rest and i + 1 < len(lines):
                rest = lines[i + 1].strip(" \t•-*")
            return rest[:300]
    return ""


@dataclass
class ImagingFinding:
    section: Section
    source: str
    impression: str
    red_flags: List[str]
    abnormal: List[str]


@dataclass
class AnalysisResult:
    lab_results: List[tuple[LabResult, str]] = field(default_factory=list)  # (result, source)
    abnormal: List[tuple[LabResult, str]] = field(default_factory=list)
    lab_conditions: List[Dict[str, str]] = field(default_factory=list)  # {condition, evidence}
    imaging: List[ImagingFinding] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    drug_alerts: List[SafetyAlert] = field(default_factory=list)  # drugs asked about
    medication_alerts: List[SafetyAlert] = field(default_factory=list)  # current medicines
    allergy_facts: str = ""
    differential: List[Dict[str, object]] = field(default_factory=list)
    documented_conditions: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    specialists: List[str] = field(default_factory=list)  # which analysers had data

    @property
    def possible_conditions(self) -> List[str]:
        names = [*self.documented_conditions, *(c["condition"] for c in self.lab_conditions)]
        names += [str(d["condition"]) for d in self.differential[:3]]
        return list(dict.fromkeys(names))

    def facts_text(self) -> str:
        """Deterministic findings handed to the Insight agent (it must not contradict them)."""
        out = []
        if self.lab_results:
            out.append(f"LAB INTERPRETER: {len(self.lab_results)} values, {len(self.abnormal)} outside reference range.")
            out += [f"- {r.line()} [{src}]" for r, src in self.abnormal[:20]]
            out += [f"- Pattern suggests {c['condition']} ({c['evidence']})" for c in self.lab_conditions[:6]]
        for f in self.imaging:
            out.append(f"RADIOLOGY ANALYZER — {f.section.title} [{f.source}]: impression: {f.impression or 'not stated'}")
            out += [f"- RED FLAG: {h}" for h in f.red_flags[:3]] + [f"- abnormal: {h}" for h in f.abnormal[:3]]
        if self.allergies or self.drug_alerts or self.medication_alerts:
            out.append("ALLERGY SAFETY: documented allergies: " + (", ".join(self.allergies) or "none recorded"))
            out += [f"- ALERT [{a.level}] {a.drug} vs {a.against}: {a.message}" for a in self.drug_alerts]
            out += [f"- current-medication alert [{a.level}] {a.drug} vs {a.against}: {a.message}" for a in self.medication_alerts[:5]]
            if self.allergy_facts and self.allergy_facts != "None.":
                out.append(self.allergy_facts)
        if self.differential:
            out.append("SYMPTOM DIFFERENTIAL (knowledge graph): " + "; ".join(
                f"{d['condition']} (score {d['score']}; confirm with {', '.join(d.get('workup', [])[:3]) or 'clinical review'})"
                for d in self.differential[:5]))
        if self.documented_conditions:
            out.append("DOCUMENTED CONDITIONS: " + ", ".join(self.documented_conditions))
        out += [f"CLINICAL INCONSISTENCY: {i}" for i in self.inconsistencies]
        return "\n".join(out) or "No structured findings."


class ClinicalAnalysisAgent:
    def __init__(self, engine: "ClinicalEngine"):
        self.engine = engine

    def run(self, ext: ExtractionResult, drugs: Sequence[str], allergens: Sequence[str],
            check_allergies: bool = True) -> AnalysisResult:
        e, kg, key = self.engine, self.engine.kg, ext.patient_key
        res = AnalysisResult()
        patient = e.patient(key)
        # ---- Lab Interpreter
        for s in ext.sections:
            if e.store.domain_of(s) != Domain.LAB and not s.lab_results:
                continue
            src = e.store.source_of(s)
            for r in s.lab_results:
                res.lab_results.append((r, src))
                if r.abnormal:
                    res.abnormal.append((r, src))
        seen = set()
        for r, _ in res.abnormal:
            test = kg.canonical_test(r.parameter)
            direction = "LOW" if r.status.value.endswith("LOW") else "HIGH"
            for cond in _indicated(test, direction):
                if cond not in seen:
                    seen.add(cond)
                    res.lab_conditions.append({"condition": cond, "evidence": f"{r.parameter} {direction.lower()}"})
        if res.lab_results:
            res.specialists.append("Lab Interpreter")
        # ---- Radiology Analyzer
        for s in ext.sections:
            if e.store.domain_of(s) != Domain.RADIOLOGY:
                continue
            res.imaging.append(ImagingFinding(section=s, source=e.store.source_of(s), impression=impression_of(s.text),
                                              red_flags=imaging_red_flags(s.text, s.title),
                                              abnormal=imaging_abnormal(s.text, s.title)))
        if res.imaging:
            res.specialists.append("Radiology Analyzer")
        # ---- Allergy Safety
        if check_allergies:
            if patient:
                res.allergies = [f"{a.allergen} ({a.severity.lower()})" for a in patient.allergies]
                res.medication_alerts = kg.medication_review(key)
            res.drug_alerts = e.safety_alerts(key, drugs)
            if drugs or allergens or (patient and patient.allergies):
                res.allergy_facts = e.allergy_facts(key, drugs, allergens)
            if res.allergies or res.drug_alerts or drugs or allergens:
                res.specialists.append("Allergy Safety")
        # ---- symptoms, documented conditions, consistency
        if ext.symptoms:
            # conditions explaining the symptoms reported right now come first, then the full symptom history
            merged = {}
            for d in [*(kg.differential(ext.new_symptoms, key) if ext.new_symptoms else []), *kg.differential(ext.symptoms, key)]:
                merged.setdefault(str(d["condition"]), d)
            res.differential = list(merged.values())
        if key:
            res.documented_conditions = [c["condition"] for c in kg.patient_conditions(key) if c["basis"] == "documented"]
        res.inconsistencies = self._consistency(ext, res)
        return res

    def _consistency(self, ext: ExtractionResult, res: AnalysisResult) -> List[str]:
        issues: List[str] = []
        if ext.new_symptoms and res.differential and (res.lab_results or res.imaging):
            candidates = [str(d["condition"]) for d in res.differential]
            supported = set(res.documented_conditions) | {c["condition"] for c in res.lab_conditions}
            for f in res.imaging:
                supported.update(self.engine.kg.conditions_in_text(f.section.text))
            if not any(c in supported for c in candidates):
                issues.append(f"Reported symptoms ({', '.join(ext.new_symptoms)}) are not explained by the available reports "
                              f"(symptom-based candidates: {', '.join(candidates[:4])}) — further evaluation needed.")
        for r, src in res.lab_results:
            if r.status_discrepancy:
                issues.append(f"{src}: report flags {r.parameter} as '{r.reported_status}' but {r.value_text} is "
                              f"{r.status.value} against the printed range {r.reference_range}.")
        return issues


def _indicated(test: Optional[str], direction: str) -> List[str]:
    return [c for t, d, c in TEST_INDICATES if t == test and d == direction] if test else []
