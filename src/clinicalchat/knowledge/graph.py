"""NetworkX medical knowledge graph: clinical ontology + patient facts + deterministic safety checks.

Vocabulary follows the course design (Week 2). Every node has a ``type``:
TEST, SYMPTOM, DIAGNOSIS, TREATMENT, ALLERGEN, MEDICATION, MEDICATION_CLASS, PATIENT. Clinical relations:

    TEST      ─CONFIRMS──────▶ DIAGNOSIS     (lab pattern with a direction, panel, or confirmatory work-up)
    SYMPTOM   ─INDICATES─────▶ DIAGNOSIS
    ALLERGEN  ─CROSS_REACTIVE▶ ALLERGEN
    TREATMENT ─TREATS────────▶ DIAGNOSIS     (medicines too)

Safety relations: ALLERGEN ─AVOID▶ MEDICATION(_CLASS), ALLERGEN ─SAFE_ALTERNATIVE▶ TREATMENT,
MEDICATION ─BELONGS_TO▶ MEDICATION_CLASS, MEDICATION ─INTERACTS▶ MEDICATION, DIAGNOSIS ─CONTRAINDICATES▶ MEDICATION.
Patient facts: HAS_ALLERGY, TAKES, HAS_CONDITION, ABNORMAL_RESULT, SUGGESTED_CONDITION, REPORTS_SYMPTOM.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from clinicalchat.domain.models import LabStatus, PatientProfile, SafetyAlert
from clinicalchat.knowledge import ontology as O

ALLERGY_RISK_TO_LEVEL = {"HIGH": "CONTRAINDICATED", "MODERATE": "MAJOR", "LOW": "MODERATE"}
NODE_TYPES = {"test": "TEST", "symptom": "SYMPTOM", "condition": "DIAGNOSIS", "treatment": "TREATMENT",
              "allergen": "ALLERGEN", "drug": "MEDICATION", "drug_class": "MEDICATION_CLASS", "patient": "PATIENT"}
CONFIRMS, INDICATES, CROSS_REACTIVE, TREATS = "CONFIRMS", "INDICATES", "CROSS_REACTIVE", "TREATS"


def nid(kind: str, name: str) -> str:
    return f"{kind}:{name}"


@lru_cache(maxsize=4096)
def _term_regex(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")


_NEGATED = re.compile(r"\b(no|not|without|denies|deny|never|nor)\b(?:\s+\w+){0,2}\s*$")


def _mentions(text: str, term: str) -> bool:
    return bool(_term_regex(term).search(text))


class MedicalKnowledgeGraph:
    def __init__(self, patients: Optional[Iterable[PatientProfile]] = None):
        self.g = nx.MultiDiGraph()
        self._build_ontology()
        self.patients: Dict[str, PatientProfile] = {}
        for p in patients or []:
            self.add_patient(p)

    # ------------------------------------------------------------------ construction
    def _node(self, kind: str, name: str, **attrs) -> str:
        key = nid(kind, name)
        if key not in self.g:
            self.g.add_node(key, kind=kind, type=NODE_TYPES[kind], label=name, **attrs)
        return key

    def _build_ontology(self) -> None:
        """Static medical knowledge from ``ontology.py``: what things are, what is unsafe, what suggests what."""
        self._add_drugs_and_allergens()
        self._add_safety_rules()
        self._add_clinical_reasoning()

    def _add_drugs_and_allergens(self) -> None:
        for drug, (classes, aliases) in O.DRUGS.items():
            d = self._node("drug", drug, aliases=aliases)
            for c in classes:
                self.g.add_edge(d, self._node("drug_class", c), rel="BELONGS_TO")
        for allergen, (cat, aliases) in O.ALLERGENS.items():
            self._node("allergen", allergen, category=cat, aliases=aliases)
        for a, b, risk, why in O.CROSS_REACTIVITY:
            for x, y in ((a, b), (b, a)) if a != b else ((a, b),):
                self.g.add_edge(nid("allergen", x), nid("allergen", y), rel=CROSS_REACTIVE, risk=risk, reason=why)

    def _add_safety_rules(self) -> None:
        """Allergen -> drugs to avoid, safe alternatives, drug-drug interactions."""
        for allergen, rules in O.ALLERGEN_DRUG_RULES.items():
            for target, kind, risk, why in rules:
                tnode = self._node("drug" if kind == "drug" else "drug_class", target)
                self.g.add_edge(nid("allergen", allergen), tnode, rel="AVOID", risk=risk, reason=why)
        for allergen, alts in O.ALLERGEN_ALTERNATIVES.items():
            for alt in alts:
                self.g.add_edge(nid("allergen", allergen), self._node("treatment", alt), rel="SAFE_ALTERNATIVE")
        for a, b, sev, effect, mgmt in O.DRUG_INTERACTIONS:
            self.g.add_edge(nid("drug", a), nid("drug", b), rel="INTERACTS", severity=sev, effect=effect, management=mgmt)
            self.g.add_edge(nid("drug", b), nid("drug", a), rel="INTERACTS", severity=sev, effect=effect, management=mgmt)

    def _add_clinical_reasoning(self) -> None:
        """Conditions, contraindications, tests, symptoms and work-up / treatment links."""
        for cond in O.CONDITION_KEYWORDS:
            self._node("condition", cond)
        for cond, rules in O.CONDITION_CONTRAINDICATIONS.items():
            for target, kind, sev, why in rules:
                tnode = self._node("drug" if kind == "drug" else "drug_class", target)
                self.g.add_edge(self._node("condition", cond), tnode, rel="CONTRAINDICATES", severity=sev, reason=why)
        for test in O.TESTS:
            self._node("test", test)
        for test, direction, cond in O.TEST_INDICATES:
            self.g.add_edge(nid("test", test), self._node("condition", cond), rel=CONFIRMS, direction=direction,
                            basis="lab pattern")
        for panel, conds in O.TEST_PANELS.items():
            for cond in conds:
                self.g.add_edge(self._node("test", panel), self._node("condition", cond), rel=CONFIRMS, basis="panel")
        for symptom, conds in O.SYMPTOMS.items():
            s = self._node("symptom", symptom)
            for cond in conds:
                self.g.add_edge(s, self._node("condition", cond), rel=INDICATES)
        for cond, (workup, treatments) in O.CONDITION_WORKUP.items():
            c = self._node("condition", cond)
            for w in workup:
                self.g.add_edge(self._node("test", w), c, rel=CONFIRMS, basis="work-up")
            for t in treatments:
                source = nid("drug", t) if t in O.DRUGS else self._node("treatment", t)
                self.g.add_edge(source, c, rel=TREATS)

    def add_patient(self, p: PatientProfile) -> None:
        self.patients[p.patient_key] = p
        pn = self._node("patient", p.patient_key, display=p.label)
        for a in p.allergies:
            allergen = self.resolve_allergen(a.allergen) or a.allergen.lower()
            an = self._node("allergen", allergen, category=a.category, aliases=[])
            self.g.add_edge(pn, an, rel="HAS_ALLERGY", severity=a.severity, reactions=", ".join(a.reactions), source=a.source)
        for med in p.medications:
            for drug in self.resolve_drugs(med):
                self.g.add_edge(pn, nid("drug", drug), rel="TAKES", evidence=med)
        for cond_text in p.conditions:
            for cond in self.conditions_in_text(cond_text):
                self.g.add_edge(pn, nid("condition", cond), rel="HAS_CONDITION", evidence=cond_text[:160])
        for lab in p.abnormal_labs:
            test = self.canonical_test(lab.parameter)
            if not test:
                continue
            direction = "LOW" if lab.status in (LabStatus.LOW, LabStatus.CRITICAL_LOW) else "HIGH"
            tn = nid("test", test)
            self.g.add_edge(pn, tn, rel="ABNORMAL_RESULT", direction=direction, evidence=lab.line())
            for _, cn, data in list(self.g.out_edges(tn, data=True)):
                if data.get("rel") == CONFIRMS and data.get("direction") == direction:
                    self.g.add_edge(pn, cn, rel="SUGGESTED_CONDITION", evidence=lab.line())
        for reported in p.symptoms:
            for symptom in self.symptoms_in_text(reported) or []:
                self.g.add_edge(pn, nid("symptom", symptom), rel="REPORTS_SYMPTOM", evidence=reported)

    def refresh_patient(self, p: PatientProfile) -> None:
        """Rebuild a patient's edges after new reports or intake data."""
        pn = nid("patient", p.patient_key)
        if pn in self.g:
            self.g.remove_node(pn)
        self.add_patient(p)

    def patient_symptoms(self, patient_key: str) -> List[str]:
        pn = nid("patient", patient_key)
        if pn not in self.g:
            return []
        return list(dict.fromkeys(self.g.nodes[v]["label"] for _, v, d in self.g.out_edges(pn, data=True)
                                  if d["rel"] == "REPORTS_SYMPTOM"))

    # ------------------------------------------------------------------ resolution helpers
    def resolve_drugs(self, text: str) -> List[str]:
        t = text.lower()
        found = []
        for drug, (_, aliases) in O.DRUGS.items():
            if any(_mentions(t, term) for term in [drug, *aliases]):
                found.append(drug)
        # "nsaid" mentioned generically -> represent with ibuprofen as class exemplar
        if not found and re.search(r"\bnsaids?\b", t):
            found.append("ibuprofen")
        return found

    def resolve_allergen(self, text: str) -> Optional[str]:
        t = text.lower().strip()
        for allergen, (_, aliases) in O.ALLERGENS.items():
            if t == allergen or t in aliases:
                return allergen
        for allergen, (_, aliases) in O.ALLERGENS.items():
            if any(_mentions(t, term) for term in [allergen, *aliases]):
                return allergen
        return None

    def allergens_in_text(self, text: str) -> List[str]:
        t = text.lower()
        return [a for a, (_, aliases) in O.ALLERGENS.items() if any(_mentions(t, term) for term in [a, *aliases])]

    def conditions_in_text(self, text: str) -> List[str]:
        t = text.lower()
        return [c for c, kws in O.CONDITION_KEYWORDS.items() if any(_mentions(t, k) for k in kws)]

    def symptoms_in_text(self, text: str) -> List[str]:
        """Canonical symptoms mentioned in free text (synonyms resolved, negated mentions such as 'no fever' skipped)."""
        t = text.lower().replace("’", "'")
        found: List[str] = []
        for term, canon in [*((s, s) for s in O.SYMPTOMS), *O.SYMPTOM_SYNONYMS.items()]:
            for m in _term_regex(term).finditer(t):
                if not _NEGATED.search(t[max(0, m.start() - 25):m.start()]):
                    found.append(canon)
                    break
        return list(dict.fromkeys(found))

    @staticmethod
    def canonical_test(parameter: str) -> Optional[str]:
        p = parameter.lower().strip()
        for test, patterns in O.TESTS.items():
            if any(re.search(pat, p) for pat in patterns):
                return test
        return None

    def drug_classes(self, drug: str) -> List[str]:
        node = nid("drug", drug)
        if node not in self.g:
            return []
        return [self.g.nodes[v]["label"] for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == "BELONGS_TO"]

    # ------------------------------------------------------------------ queries
    def cross_reactive(self, allergen: str) -> List[Dict[str, str]]:
        a = self.resolve_allergen(allergen) or allergen.lower()
        node = nid("allergen", a)
        if node not in self.g:
            return []
        out = []
        for _, v, d in self.g.out_edges(node, data=True):
            if d["rel"] == CROSS_REACTIVE:
                out.append({"allergen": self.g.nodes[v]["label"], "risk": d["risk"], "reason": d["reason"], "type": "allergen"})
            elif d["rel"] == "AVOID":
                out.append({"allergen": self.g.nodes[v]["label"], "risk": d["risk"], "reason": d["reason"], "type": "drug"})
        order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
        return sorted(out, key=lambda x: order.get(x["risk"], 3))

    def safe_alternatives(self, allergen: str) -> List[str]:
        a = self.resolve_allergen(allergen) or allergen.lower()
        node = nid("allergen", a)
        if node not in self.g:
            return []
        return [self.g.nodes[v]["label"] for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == "SAFE_ALTERNATIVE"]

    def patient_drugs(self, patient_key: str) -> List[Tuple[str, str]]:
        pn = nid("patient", patient_key)
        if pn not in self.g:
            return []
        return [(self.g.nodes[v]["label"], d.get("evidence", "")) for _, v, d in self.g.out_edges(pn, data=True) if d["rel"] == "TAKES"]

    def patient_conditions(self, patient_key: str) -> List[Dict[str, str]]:
        pn = nid("patient", patient_key)
        if pn not in self.g:
            return []
        out: Dict[str, Dict[str, str]] = {}
        for _, v, d in self.g.out_edges(pn, data=True):
            if d["rel"] in ("HAS_CONDITION", "SUGGESTED_CONDITION"):
                label = self.g.nodes[v]["label"]
                entry = out.setdefault(label, {"condition": label, "basis": "documented" if d["rel"] == "HAS_CONDITION" else "lab-suggested", "evidence": []})
                if d["rel"] == "HAS_CONDITION":
                    entry["basis"] = "documented"
                entry["evidence"].append(d.get("evidence", ""))
        return list(out.values())

    def check_drug(self, drug_text: str, allergies: Sequence[str] = (), current_drugs: Sequence[str] = (),
                   conditions: Sequence[str] = ()) -> List[SafetyAlert]:
        """All allergy, interaction and condition alerts for a proposed drug (deduplicated, most severe first)."""
        alerts: List[SafetyAlert] = []
        for drug in self.resolve_drugs(drug_text) or [drug_text.lower()]:
            classes = set(self.drug_classes(drug))
            for allergy in allergies:
                alerts += self._allergy_alerts(drug, classes, allergy)
            for other in current_drugs:
                alerts += self._interaction_alerts(drug, other)
            for cond in conditions:
                alerts += self._condition_alerts(drug, classes, cond)
        uniq = {(a.level, a.category, a.drug, a.against, a.message): a for a in alerts}
        return sorted(uniq.values(), key=lambda a: a.rank)

    def _edges(self, node: str, rel: str):
        return [(v, d) for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == rel]

    def _allergy_alerts(self, drug: str, classes: set, allergy: str) -> List[SafetyAlert]:
        allergen = self.resolve_allergen(allergy) or allergy.lower()
        alts = self.safe_alternatives(allergen)
        if allergen == drug:
            return [SafetyAlert(level="CONTRAINDICATED", category="allergy", drug=drug, against=allergy,
                                message=f"Patient has a documented {allergy} allergy",
                                management="Do not administer; choose an alternative: " + ", ".join(alts) if alts
                                else "Do not administer")]
        anode = nid("allergen", allergen)
        if anode not in self.g:
            return []
        out = []
        for v, d in self._edges(anode, "AVOID"):
            target = self.g.nodes[v]["label"]
            if target == drug or target in classes:
                out.append(SafetyAlert(
                    level=ALLERGY_RISK_TO_LEVEL.get(d["risk"], "MODERATE"), category="allergy", drug=drug, against=allergy,
                    message=f"{allergy} allergy -> {target}: {d['reason']}",
                    management=("Alternatives: " + ", ".join(alts)) if alts else "Review with prescriber"))
        return out

    def _interaction_alerts(self, drug: str, other: str) -> List[SafetyAlert]:
        if other == drug:
            return []
        return [SafetyAlert(level=d["severity"], category="interaction", drug=drug, against=other,
                            message=d["effect"], management=d["management"])
                for v, d in self._edges(nid("drug", drug), "INTERACTS") if self.g.nodes[v]["label"] == other]

    def _condition_alerts(self, drug: str, classes: set, cond: str) -> List[SafetyAlert]:
        cnode = nid("condition", cond)
        if cnode not in self.g:
            return []
        if drug in self.treatments_for(cond):  # e.g. aspirin is a treatment for MI, not a contraindication
            return []  # the drug treats this condition
        return [SafetyAlert(level=d["severity"], category="condition", drug=drug, against=cond, message=d["reason"],
                            management="Review risk/benefit; consider alternatives")
                for v, d in self._edges(cnode, "CONTRAINDICATES") if self.g.nodes[v]["label"] in classes | {drug}]

    def check_drug_for_patient(self, drug_text: str, patient_key: str) -> List[SafetyAlert]:
        p = self.patients.get(patient_key)
        if not p:
            return self.check_drug(drug_text)
        conds = self.patient_conditions(patient_key)
        alerts = self.check_drug(
            drug_text,
            allergies=[a.allergen for a in p.allergies],
            current_drugs=[d for d, _ in self.patient_drugs(patient_key)],
            conditions=[c["condition"] for c in conds],
        )
        evidence = {c["condition"]: f"{c['basis']}: " + "; ".join(e for e in c["evidence"] if e)[:200] for c in conds}
        for a in alerts:
            if a.category == "condition":
                a.evidence = evidence.get(a.against, "")
        return alerts

    def medication_review(self, patient_key: str) -> List[SafetyAlert]:
        """Interactions and condition contraindications among a patient's current medications."""
        drugs = [d for d, _ in self.patient_drugs(patient_key)]
        alerts: Dict[tuple, SafetyAlert] = {}
        for drug in drugs:
            for a in self.check_drug_for_patient(drug, patient_key):
                key = (a.category, tuple(sorted([a.drug, a.against])), a.message)
                alerts.setdefault(key, a)
        return sorted(alerts.values(), key=lambda a: a.rank)

    def differential(self, symptoms: Sequence[str], patient_key: Optional[str] = None) -> List[Dict[str, object]]:
        scores: Dict[str, Dict[str, object]] = {}
        for s in symptoms:
            sn = nid("symptom", s)
            if sn not in self.g:
                continue
            for _, v, d in self.g.out_edges(sn, data=True):
                if d["rel"] == INDICATES:
                    e = scores.setdefault(self.g.nodes[v]["label"], {"condition": self.g.nodes[v]["label"], "score": 0, "support": []})
                    e["score"] += 1
                    e["support"].append(f"symptom: {s}")
        if patient_key:
            for c in self.patient_conditions(patient_key):
                if c["condition"] in scores:
                    scores[c["condition"]]["score"] += 2
                    scores[c["condition"]]["support"].append(f"patient record ({c['basis']})")
        for e in scores.values():
            workup, treat = O.CONDITION_WORKUP.get(e["condition"], ([], []))
            e["workup"], e["treatment"] = workup, treat
        return sorted(scores.values(), key=lambda e: -e["score"])

    # ------------------------------------------------------------------ course query functions (Week 2)
    def find_diagnoses_for_symptoms(self, symptoms: Sequence[str]) -> List[Dict[str, object]]:
        """Diagnoses INDICATED by the symptoms, ranked by how many symptoms point to each (free text is accepted)."""
        canonical = list(dict.fromkeys(c for s in symptoms for c in (self.symptoms_in_text(s) or [s.lower().strip()])))
        return [{"diagnosis": d["condition"], "matching_symptoms": d["score"],
                 "symptoms": [x.split(": ", 1)[1] for x in d["support"]], "confirm_with": d["workup"]}
                for d in self.differential(canonical)]

    def get_cross_reactive_allergens(self, allergen: str) -> List[str]:
        """Allergens linked by CROSS_REACTIVE (course function); medicines to avoid come from ``cross_reactive``."""
        return [x["allergen"] for x in self.cross_reactive(allergen) if x["type"] == "allergen"]

    def tests_confirming(self, diagnosis: str) -> List[str]:
        node = nid("condition", diagnosis)
        if node not in self.g:
            return []
        return sorted({self.g.nodes[u]["label"] for u, _, d in self.g.in_edges(node, data=True)
                       if d["rel"] == CONFIRMS})

    def treatments_for(self, diagnosis: str) -> List[str]:
        node = nid("condition", diagnosis)
        if node not in self.g:
            return []
        return [self.g.nodes[u]["label"] for u, _, d in self.g.in_edges(node, data=True) if d["rel"] == TREATS]

    def allergy_safety_card(self, allergen: str) -> Dict[str, object]:
        """Everything the "Allergy Safety Check" needs about one allergen (knowledge graph only, no patient data)."""
        name = self.resolve_allergen(allergen)
        node = nid("allergen", name) if name else None
        related = self.cross_reactive(name) if name else []
        return {"query": allergen, "allergen": name,
                "category": self.g.nodes[node].get("category", "") if node in self.g else "",
                "cross_reactive_allergens": [x for x in related if x["type"] == "allergen"],
                "avoid_medicines": [x for x in related if x["type"] == "drug"],
                "safe_alternatives": self.safe_alternatives(name) if name else []}

    def known_allergens(self) -> List[str]:
        return sorted(O.ALLERGENS)

    def to_mermaid(self, relations: Sequence[str] = (CONFIRMS, INDICATES, CROSS_REACTIVE, TREATS),
                   only: Optional[Iterable[str]] = None) -> str:
        """The course relations as a Mermaid flowchart (used to draw docs/diagrams/09-knowledge-graph.png)."""
        keep = set(only) if only else None
        ids: Dict[str, str] = {}
        drawn: Set[Tuple[str, str, str]] = set()
        lines = ["flowchart LR"]
        for u, v, d in self.g.edges(data=True):
            if d["rel"] not in relations or (keep and not ({u, v} <= keep)):
                continue
            key = (*sorted((u, v)), d["rel"]) if d["rel"] == CROSS_REACTIVE else (u, v, d["rel"])  # symmetric / multi-edges once
            if key in drawn:
                continue
            drawn.add(key)
            for n in (u, v):
                if n not in ids:
                    ids[n] = f"n{len(ids)}"
                    node = self.g.nodes[n]
                    lines.append(f'  {ids[n]}["{node["type"]}: {node["label"]}"]:::{node["type"].lower()}')
            lines.append(f"  {ids[u]} -->|{d['rel']}| {ids[v]}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            counts[d["kind"]] = counts.get(d["kind"], 0) + 1
        counts["nodes"] = self.g.number_of_nodes()
        counts["edges"] = self.g.number_of_edges()
        return counts
