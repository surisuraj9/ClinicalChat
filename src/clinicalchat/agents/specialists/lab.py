"""Lab Interpreter: every value against its printed reference range; abnormal patterns -> possible conditions
(knowledge-graph rule TEST_INDICATES, e.g. low haemoglobin -> anaemia)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from clinicalchat.domain.models import LabResult
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.knowledge.ontology import TEST_INDICATES

SourcedLab = Tuple[LabResult, str]  # (result, source file)


@dataclass
class LabFindings:
    results: List[SourcedLab] = field(default_factory=list)
    abnormal: List[SourcedLab] = field(default_factory=list)
    conditions: List[Dict[str, str]] = field(default_factory=list)  # {condition, evidence}


def indicated(test: Optional[str], direction: str) -> List[str]:
    return [c for t, d, c in TEST_INDICATES if t == test and d == direction] if test else []


def interpret_labs(results: Iterable[SourcedLab]) -> LabFindings:
    out = LabFindings(results=list(results))
    out.abnormal = [(r, src) for r, src in out.results if r.abnormal]
    seen = set()
    for r, _ in out.abnormal:
        direction = "LOW" if r.status.value.endswith("LOW") else "HIGH"
        for cond in indicated(MedicalKnowledgeGraph.canonical_test(r.parameter), direction):
            if cond not in seen:
                seen.add(cond)
                out.conditions.append({"condition": cond, "evidence": f"{r.parameter} {direction.lower()}"})
    return out
