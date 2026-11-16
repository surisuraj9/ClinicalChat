"""Risk vocabulary shared by the Risk Detection agent, the answers and the UI: levels, badges and patient guidance."""

from __future__ import annotations

from typing import Iterable, List

from pydantic import BaseModel, Field

LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
ESCALATING_LEVELS = ("HIGH", "CRITICAL")
CLINICAL_CATEGORIES = {"lab", "imaging", "symptom", "pattern"}
LEVEL_TO_URGENCY = {"LOW": "ROUTINE", "MODERATE": "SOON", "HIGH": "URGENT", "CRITICAL": "EMERGENCY"}
LEVEL_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
LEVEL_BADGE = {level: f"{icon} {level}" for level, icon in LEVEL_ICON.items()}

PATIENT_GUIDANCE = {
    "CRITICAL": "🚑 **This could be an emergency. Call your local emergency number (e.g. 112 / 108 / 911) or go to the "
                "nearest emergency department now.** Do not wait for an online reply.",
    "HIGH": "🏥 **Please see a doctor today** (urgent care or your GP). If symptoms get worse, call emergency services.",
    "MODERATE": "📅 **Book an appointment with your doctor in the next few days.** Seek care sooner if you feel worse.",
    "LOW": "🗓️ **No urgent warning signs found.** Discuss these results at your next routine visit, or sooner if "
           "you feel unwell.",
}


def highest(levels: Iterable[str]) -> str:
    """The most severe of the given levels (unknown values count as LOW)."""
    return max(levels, key=lambda x: LEVELS.index(x) if x in LEVELS else 0, default="LOW")


def bump(level: str) -> str:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


class RiskIndicator(BaseModel):
    indicator: str
    level: str = "LOW"
    category: str = "lab"  # lab | medication | allergy | imaging | symptom | pattern | consistency | extraction | ai_specialist
    source: str = ""
    escalate: bool = False


class RiskAssessment(BaseModel):
    risk_level: str = "LOW"
    urgency: str = "ROUTINE"
    score: int = 0
    escalate: bool = False
    escalation_reasons: List[str] = Field(default_factory=list)
    indicators: List[RiskIndicator] = Field(default_factory=list)

    @property
    def clinical_level(self) -> str:
        """Risk from the patient's condition (labs, imaging, symptoms) — drives patient guidance; a question about
        a contraindicated drug is a safety alert, not a sign that the patient is unwell."""
        return highest(i.level for i in self.indicators if i.category in CLINICAL_CATEGORIES)
