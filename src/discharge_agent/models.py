"""Pydantic models for discharge summary structured output."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FlagLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Flag(BaseModel):
    level: FlagLevel
    section: str
    message: str


class MedicationChange(BaseModel):
    medication: str
    change_type: str  # "added" | "stopped" | "changed" | "continued"
    admission_dose: Optional[str] = None
    discharge_dose: Optional[str] = None
    reason: Optional[str] = None
    flagged: bool = False
    flag_note: Optional[str] = None


class PendingResult(BaseModel):
    test_name: str
    ordered_date: Optional[str] = None
    note: str


class DischargeSummary(BaseModel):
    # Demographics
    patient_name: Optional[str] = Field(None, description="Patient full name")
    date_of_birth: Optional[str] = None
    mrn: Optional[str] = Field(None, description="Medical Record Number")
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None

    # Clinical
    principal_diagnosis: Optional[str] = None
    secondary_diagnoses: list[str] = Field(default_factory=list)
    hospital_course: Optional[str] = None
    procedures: list[str] = Field(default_factory=list)

    # Medications
    admission_medications: list[str] = Field(default_factory=list)
    discharge_medications: list[str] = Field(default_factory=list)
    medication_changes: list[MedicationChange] = Field(default_factory=list)

    # Safety
    allergies: list[str] = Field(default_factory=list)
    discharge_condition: Optional[str] = None
    follow_up_instructions: list[str] = Field(default_factory=list)
    pending_results: list[PendingResult] = Field(default_factory=list)

    # Meta
    flags: list[Flag] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    is_draft: bool = True
    generation_note: str = "DRAFT — For clinician review only. Not for clinical use."

    def add_flag(self, level: FlagLevel, section: str, message: str) -> None:
        self.flags.append(Flag(level=level, section=section, message=message))

    def mark_missing(self, field: str) -> None:
        if field not in self.missing_fields:
            self.missing_fields.append(field)
