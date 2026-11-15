"""NetworkX medical knowledge graph: clinical ontology + patient facts + deterministic safety checks."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from clinicalchat import kg_ontology as O
from clinicalchat.models import LabStatus, PatientProfile, SafetyAlert

ALLERGY_RISK_TO_LEVEL = {"HIGH": "CONTRAINDICATED", "MODERATE": "MAJOR", "LOW": "MODERATE"}


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
            self.g.add_node(key, kind=kind, label=name, **attrs)
        return key

    def _build_ontology(self) -> None:
        for drug, (classes, aliases) in O.DRUGS.items():
            d = self._node("drug", drug, aliases=aliases)
            for c in classes:
                self.g.add_edge(d, self._node("drug_class", c), rel="BELONGS_TO")
        for allergen, (cat, aliases) in O.ALLERGENS.items():
            self._node("allergen", allergen, category=cat, aliases=aliases)
        for a, b, risk, why in O.CROSS_REACTIVITY:
            for x, y in ((a, b), (b, a)) if a != b else ((a, b),):
                self.g.add_edge(nid("allergen", x), nid("allergen", y), rel="CROSS_REACTS", risk=risk, reason=why)
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
        for cond in O.CONDITION_KEYWORDS:
            self._node("condition", cond)
        for cond, rules in O.CONDITION_CONTRAINDICATIONS.items():
            for target, kind, sev, why in rules:
                tnode = self._node("drug" if kind == "drug" else "drug_class", target)
                self.g.add_edge(self._node("condition", cond), tnode, rel="CONTRAINDICATES", severity=sev, reason=why)
        for test in O.TESTS:
            self._node("test", test)
        for test, direction, cond in O.TEST_INDICATES:
            self.g.add_edge(nid("test", test), self._node("condition", cond), rel="INDICATES", direction=direction)
        for symptom, conds in O.SYMPTOMS.items():
            s = self._node("symptom", symptom)
            for cond in conds:
                self.g.add_edge(s, self._node("condition", cond), rel="SUGGESTS")
        for cond, (workup, treatments) in O.CONDITION_WORKUP.items():
            c = self._node("condition", cond)
            for w in workup:
                self.g.add_edge(c, self._node("test", w) if w in O.TESTS else self._node("treatment", w), rel="CONFIRMED_BY")
            for t in treatments:
                target = nid("drug", t) if t in O.DRUGS else self._node("treatment", t)
                self.g.add_edge(c, target, rel="TREATED_WITH")

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
                if data.get("rel") == "INDICATES" and data.get("direction") == direction:
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
            if d["rel"] == "CROSS_REACTS":
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
        alerts: List[SafetyAlert] = []
        for drug in self.resolve_drugs(drug_text) or [drug_text.lower()]:
            classes = set(self.drug_classes(drug))
            for allergy in allergies:
                allergen = self.resolve_allergen(allergy) or allergy.lower()
                if allergen == drug:
                    alerts.append(SafetyAlert(level="CONTRAINDICATED", category="allergy", drug=drug, against=allergy,
                                              message=f"Patient has a documented {allergy} allergy",
                                              management="Do not administer; choose an alternative: " + ", ".join(self.safe_alternatives(allergen)) if self.safe_alternatives(allergen) else "Do not administer"))
                    continue
                anode = nid("allergen", allergen)
                if anode not in self.g:
                    continue
                for _, v, d in self.g.out_edges(anode, data=True):
                    if d["rel"] != "AVOID":
                        continue
                    target = self.g.nodes[v]["label"]
                    if target == drug or target in classes:
                        alts = self.safe_alternatives(allergen)
                        alerts.append(SafetyAlert(
                            level=ALLERGY_RISK_TO_LEVEL.get(d["risk"], "MODERATE"), category="allergy", drug=drug, against=allergy,
                            message=f"{allergy} allergy -> {target}: {d['reason']}",
                            management=("Alternatives: " + ", ".join(alts)) if alts else "Review with prescriber",
                        ))
            for other in current_drugs:
                if other == drug:
                    continue
                for _, v, d in self.g.out_edges(nid("drug", drug), data=True):
                    if d["rel"] == "INTERACTS" and self.g.nodes[v]["label"] == other:
                        alerts.append(SafetyAlert(level=d["severity"], category="interaction", drug=drug, against=other,
                                                  message=d["effect"], management=d["management"]))
            for cond in conditions:
                cnode = nid("condition", cond)
                if cnode not in self.g:
                    continue
                indicated = {self.g.nodes[v]["label"] for _, v, d in self.g.out_edges(cnode, data=True) if d["rel"] == "TREATED_WITH"}
                if drug in indicated:
                    continue
                for _, v, d in self.g.out_edges(cnode, data=True):
                    if d["rel"] == "CONTRAINDICATES" and self.g.nodes[v]["label"] in classes | {drug}:
                        alerts.append(SafetyAlert(level=d["severity"], category="condition", drug=drug, against=cond,
                                                  message=d["reason"], management="Review risk/benefit; consider alternatives"))
        uniq = {(a.level, a.category, a.drug, a.against, a.message): a for a in alerts}
        return sorted(uniq.values(), key=lambda a: a.rank)

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
                if d["rel"] == "SUGGESTS":
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

    def explain_path(self, source: str, target: str) -> List[str]:
        ug = nx.Graph(self.g)
        try:
            path = nx.shortest_path(ug, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        steps = []
        for a, b in zip(path, path[1:]):
            data = self.g.get_edge_data(a, b) or self.g.get_edge_data(b, a) or {}
            rel = next(iter(data.values()), {}).get("rel", "RELATED") if data else "RELATED"
            steps.append(f"{self.g.nodes[a]['label']} -[{rel}]- {self.g.nodes[b]['label']}")
        return steps

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            counts[d["kind"]] = counts.get(d["kind"], 0) + 1
        counts["edges"] = self.g.number_of_edges()
        return counts
