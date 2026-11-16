"""Core data models shared across ingestion, agents and UI."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Domain(str, Enum):
    LAB = "lab"
    RADIOLOGY = "radiology"
    ALLERGY = "allergy"
    PATIENT_RECORD = "patient_record"
    REFERENCE = "reference"


class LabStatus(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL_LOW = "CRITICAL_LOW"
    CRITICAL_HIGH = "CRITICAL_HIGH"
    UNKNOWN = "UNKNOWN"


class LabResult(BaseModel):
    parameter: str
    value_text: str
    value: Optional[float] = None
    unit: str = ""
    reference_range: str = ""
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    status: LabStatus = LabStatus.UNKNOWN
    reported_status: str = ""
    status_discrepancy: bool = False
    origin: str = "parser"

    @property
    def abnormal(self) -> bool:
        return self.status not in (LabStatus.NORMAL, LabStatus.UNKNOWN)

    def line(self) -> str:
        rng = f" (ref {self.reference_range})" if self.reference_range else ""
        rep = f" [reported: {self.reported_status}]" if self.reported_status else ""
        return f"{self.parameter}: {self.value_text}{rng} -> {self.status.value}{rep}"


class Section(BaseModel):
    section_id: str
    doc_id: str
    title: str
    text: str
    patient_key: Optional[str] = None
    patient_name: Optional[str] = None
    fields: Dict[str, str] = Field(default_factory=dict)
    lab_results: List[LabResult] = Field(default_factory=list)
    report_date: Optional[str] = None  # ISO date (YYYY-MM-DD) when the report states one


class ClinicalDocument(BaseModel):
    doc_id: str
    filename: str
    relative_path: str
    category: str
    domain: Domain
    title: str
    sections: List[Section] = Field(default_factory=list)
    origin: str = "dataset"  # dataset | upload
    extraction_method: str = "text"  # text | ocr
    uploaded_by: str = ""
    uploaded_at: str = ""



class AllergyRecord(BaseModel):
    allergen: str
    category: str
    severity: str = "UNKNOWN"  # MILD | MODERATE | SEVERE | CRITICAL | UNKNOWN
    reactions: List[str] = Field(default_factory=list)
    emergency_medication: str = ""
    restrictions: str = ""
    alternatives: str = ""
    evidence: List[str] = Field(default_factory=list)
    source: str = ""


class PatientProfile(BaseModel):
    patient_key: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    identifiers: List[str] = Field(default_factory=list)
    section_ids: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    domains: List[Domain] = Field(default_factory=list)
    allergies: List[AllergyRecord] = Field(default_factory=list)
    negative_allergy_tests: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    abnormal_labs: List[LabResult] = Field(default_factory=list)
    imaging_studies: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        bits = [str(self.age) if self.age is not None else "", (self.gender or "")[:1].upper()]
        meta = " ".join(b for b in bits if b)
        return f"{self.name} ({meta})" if meta else self.name


class SafetyAlert(BaseModel):
    level: str  # CONTRAINDICATED | MAJOR | MODERATE | MINOR
    category: str  # allergy | interaction | condition
    drug: str
    against: str
    message: str
    management: str = ""
    evidence: str = ""

    @property
    def rank(self) -> int:
        return {"CONTRAINDICATED": 0, "MAJOR": 1, "MODERATE": 2, "MINOR": 3}.get(self.level, 4)
