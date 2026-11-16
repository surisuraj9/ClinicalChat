"""The course's AI specialist agents (Week 4): 🧪 Lab Interpreter, 🩻 Radiology Analyzer, 🛡️ Allergy Safety.

Each specialist is ONE LangChain call (prompt | gpt-4.1-mini — the course used gpt-4o-mini, which measured ~2.5×
slower here; OPENAI_SPECIALIST_MODEL switches it) whose reply is parsed into the course's Pydantic models
(``LabInterpretation``, ``RadiologyFinding``, ``AllergyAssessment`` — field for field as in the notebook), using
``PydanticOutputParser.get_format_instructions()`` like the course does. A report usually holds several tests /
studies / allergies, so each agent returns a small list of them.

They run inside the 🔬 Clinical Analysis step (services/specialist_panel.py) and never replace the rule-based
analysers next to them (lab.py / radiology.py / allergy.py): reference ranges, red flags and knowledge-graph safety
checks stay deterministic, and an AI "stat / emergency" urgency can only RAISE the risk level (agents/risk.py).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from clinicalchat.agents.llm.base import AgentResult, LLMAgent
from clinicalchat.agents.llm.prompts import GROUND_RULES
from clinicalchat.domain.roles import AGENT_LABELS, AgentName

# --- course output models (Week 4.2 – 4.4) ---------------------------------------------------------------------


class LabInterpretation(BaseModel):
    """Structured output for lab result interpretation."""
    test_name: str = Field(description="Name of the lab test")
    result_value: str = Field(description="The actual result value with units")
    status: str = Field(description="Status: NORMAL, LOW, HIGH, CRITICAL_LOW, CRITICAL_HIGH")
    clinical_significance: str = Field(description="What this result means clinically")
    potential_causes: List[str] = Field(description="Possible causes for abnormal results")
    recommended_actions: List[str] = Field(description="Suggested follow-up actions")
    urgency: str = Field(description="Urgency: routine, urgent, stat, emergency")
    confidence: float = Field(ge=0.0, le=1.0, description="Interpretation confidence (0-1)")


class RadiologyFinding(BaseModel):
    """Structured output for radiology interpretation (Indication -> Technique -> Findings -> Impression)."""
    study_type: str = Field(description="Type of imaging study (X-ray, CT, MRI, Ultrasound)")
    body_region: str = Field(description="Body region examined")
    findings: List[str] = Field(description="Key findings from the study")
    impression: str = Field(description="Overall impression/diagnosis")
    differential_diagnoses: List[str] = Field(description="Possible diagnoses to consider")
    recommendations: List[str] = Field(description="Recommended follow-up actions")
    urgency: str = Field(description="Urgency of findings: routine, urgent, stat")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in interpretation")


class AllergyAssessment(BaseModel):
    """Structured output for allergy safety assessment."""
    allergen: str = Field(description="The allergen identified")
    category: str = Field(description="Category: Food, Medication, Environmental, Skin")
    severity: str = Field(description="Severity: MILD, MODERATE, SEVERE, CRITICAL")
    reactions: List[str] = Field(description="Known reactions to this allergen")
    cross_reactivities: List[str] = Field(description="Related allergens to avoid")
    safe_alternatives: List[str] = Field(description="Safe alternatives if applicable")
    emergency_protocol: str = Field(description="Emergency response if exposure occurs")
    confidence: float = Field(ge=0.0, le=1.0, description="Assessment confidence")


class LabPanel(BaseModel):
    interpretations: List[LabInterpretation] = Field(
        default_factory=list, description="One entry per clinically important test (critical/abnormal first), at most 5")


class RadiologyReview(BaseModel):
    studies: List[RadiologyFinding] = Field(default_factory=list, description="One entry per imaging study, at most 3")


class AllergyReview(BaseModel):
    assessments: List[AllergyAssessment] = Field(
        default_factory=list, description="One entry per documented allergen (most severe first), at most 4")


# --- prompts (course wording + this project's ground rules) ----------------------------------------------------

LAB_SYSTEM = """You are the LAB INTERPRETER agent — an expert clinical laboratory medicine specialist.
Interpret lab results and provide clinical significance, potential causes, and recommendations.
Each line gives the value, the laboratory's own reference range and the flag computed by validated code: use that
range and keep that flag (NORMAL / LOW / HIGH; CRITICAL_LOW / CRITICAL_HIGH only for panic values).

Reference ranges when a line has none:
- Hemoglobin: Male 13.5-17.5 g/dL, Female 12.0-16.0 g/dL
- WBC: 4,500-11,000 /uL
- Platelet: 150,000-400,000 /uL
- Glucose (fasting): 70-100 mg/dL
- Creatinine: 0.7-1.3 mg/dL
- BUN: 7-20 mg/dL
- AST: 10-40 U/L, ALT: 7-56 U/L
- TSH: 0.4-4.0 mIU/L
- Free T4: 0.8-1.8 ng/dL
Critical (panic) values: Hemoglobin <7 or >20 g/dL; Glucose <40 or >500 mg/dL; Potassium <2.5 or >6.5 mEq/L;
Sodium <120 or >160 mEq/L; WBC <2,000 or >30,000 /uL; Platelets <50,000 or >1,000,000 /uL.
Urgency: routine, urgent, stat, emergency. Interpret at most 5 tests, critical and abnormal first.
Keep each text field under 25 words and each list to at most 3 items."""

RADIOLOGY_SYSTEM = """You are the RADIOLOGY ANALYZER agent — an expert radiologist specializing in diagnostic imaging.
Analyze radiology reports and provide findings, differential diagnoses, and recommendations.
Flag urgent findings requiring immediate attention.

For each study type, consider:
- X-ray: Lung fields, heart size, bones, soft tissues
- CT: Detailed anatomy, contrast patterns, masses
- MRI: Soft tissue detail, brain/spine evaluation
- Ultrasound: Real-time imaging, Doppler flow
Urgency: routine, urgent, stat. One entry per study (at most 3).
Keep each text field under 30 words and each list to at most 4 items."""

ALLERGY_SYSTEM = """You are the ALLERGY SAFETY agent — an allergy and immunology specialist.
Assess allergy severity, identify cross-reactive allergens, and provide safety recommendations.
Patient safety is the top priority.

Known cross-reactivities:
- Peanut: tree nuts, legumes, lupin
- Penicillin: amoxicillin, cephalosporins (10% cross-react)
- Latex: banana, avocado, kiwi, chestnut
- Shellfish: shrimp, crab, lobster (crustaceans cross-react)
- Aspirin/NSAIDs: ibuprofen, naproxen
KNOWLEDGE GRAPH facts in the input were computed by validated code: never contradict them.
One entry per documented allergen (at most 4), most severe first.
Keep each text field under 30 words and each list to at most 4 items."""

SPECIALIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}\n\n{rules}\n\n{format_instructions}"),
    ("human", "{task}:\n\n{report}"),
])

_JSON = re.compile(r"\{.*\}", re.S)
# AI urgency -> the rule-based risk level it may raise the case to
URGENCY_RISK = {"emergency": "HIGH", "stat": "HIGH", "urgent": "MODERATE"}


def _fractions(value: Any) -> Any:
    """Models sometimes write confidence as 85 or "85%"; the course schema wants 0..1."""
    if isinstance(value, dict):
        out = {k: _fractions(v) for k, v in value.items()}
        conf = out.get("confidence")
        if conf is not None:
            try:
                number = float(str(conf).strip("%"))
                out["confidence"] = max(0.0, min(1.0, number / (100 if number > 1 else 1)))
            except ValueError:
                out["confidence"] = 0.5
        return out
    if isinstance(value, list):
        return [_fractions(v) for v in value]
    return value


class SpecialistAgent(LLMAgent):
    """One course specialist: report text in, a list of course Pydantic objects out."""

    system: str
    task: str
    text_field = ""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, SPECIALIST_PROMPT)
        self.parser = PydanticOutputParser(pydantic_object=self.output_model)

    def _inputs(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        return {"system": self.system, "rules": GROUND_RULES, "task": self.task,
                "format_instructions": self.parser.get_format_instructions(), **variables}

    def _parse(self, text: str) -> Tuple[BaseModel, bool]:
        try:
            return self.parser.parse(text), True
        except Exception:  # lenient second chance: confidence as a percentage, text around the JSON
            match = _JSON.search(text)
            try:
                return self.output_model.model_validate(_fractions(json.loads(match.group(0)) if match else {})), True
            except (json.JSONDecodeError, ValidationError):
                return self.output_model(), False

    def run(self, report: str, config: Optional[dict] = None) -> AgentResult:
        result = self._invoke({"report": report}, config)
        if not result.error and not result.parsed:
            result.error = "the AI reply did not match the course schema"
        return result


class LabInterpreterAgent(SpecialistAgent):
    name = AgentName.LAB_INTERPRETER
    output_model = LabPanel
    system = LAB_SYSTEM
    task = "Interpret this lab report"


class RadiologyAnalyzerAgent(SpecialistAgent):
    name = AgentName.RADIOLOGY_ANALYZER
    output_model = RadiologyReview
    system = RADIOLOGY_SYSTEM
    task = "Analyze this radiology report"


class AllergySafetyAgent(SpecialistAgent):
    name = AgentName.ALLERGY_SAFETY
    output_model = AllergyReview
    system = ALLERGY_SYSTEM
    task = "Assess this allergy documentation"


# --- the combined result attached to the Clinical Analysis ---------------------------------------------------------


class SpecialistReport(BaseModel):
    """What the AI specialists said this turn (only those whose record type the user may see and that had data)."""
    lab: Optional[LabPanel] = None
    radiology: Optional[RadiologyReview] = None
    allergy: Optional[AllergyReview] = None
    errors: Dict[str, str] = Field(default_factory=dict)  # agent label -> error
    timings: Dict[str, str] = Field(default_factory=dict)  # agent label -> "2.1s" / "cached answer"
    _results: List[AgentResult] = PrivateAttr(default_factory=list)  # raw calls, for the audit log (not stored)

    @property
    def results(self) -> List[AgentResult]:
        return self._results

    @property
    def ran(self) -> List[str]:
        return list(self.timings)

    def urgent(self) -> List[Tuple[str, str, str]]:
        """(risk level, finding, agent label) for every AI finding marked urgent / stat / emergency."""
        out = []
        for item in (self.lab.interpretations if self.lab else []):
            level = URGENCY_RISK.get(item.urgency.strip().lower())
            if level:
                out.append((level, f"{item.test_name} {item.result_value} — {item.clinical_significance}",
                            AGENT_LABELS[AgentName.LAB_INTERPRETER]))
        for study in (self.radiology.studies if self.radiology else []):
            level = URGENCY_RISK.get(study.urgency.strip().lower())
            if level:
                out.append((level, f"{study.study_type} {study.body_region}: {study.impression}",
                            AGENT_LABELS[AgentName.RADIOLOGY_ANALYZER]))
        return out

    def facts_lines(self) -> List[str]:
        """Compact text for the Insight agent's prompt and the case hand-over."""
        out: List[str] = []
        for i in (self.lab.interpretations if self.lab else []):
            out.append(f"AI LAB INTERPRETER: {i.test_name} {i.result_value} [{i.status}, {i.urgency}] — "
                       f"{i.clinical_significance} Actions: {'; '.join(i.recommended_actions[:3])}")
        for s in (self.radiology.studies if self.radiology else []):
            out.append(f"AI RADIOLOGY ANALYZER: {s.study_type} {s.body_region} [{s.urgency}] — {s.impression} "
                       f"Differential: {', '.join(s.differential_diagnoses[:3])}")
        for a in (self.allergy.assessments if self.allergy else []):
            out.append(f"AI ALLERGY SAFETY: {a.allergen} ({a.category}, {a.severity}) — avoid "
                       f"{', '.join(a.cross_reactivities[:4]) or 'n/a'}; alternatives {', '.join(a.safe_alternatives[:3]) or 'n/a'}")
        return out


SPECIALIST_AGENTS: Sequence[Type[SpecialistAgent]] = (LabInterpreterAgent, RadiologyAnalyzerAgent, AllergySafetyAgent)
