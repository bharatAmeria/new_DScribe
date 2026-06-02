"""LangGraph node implementations for the discharge summary agent."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .models import DischargeSummary, Flag, FlagLevel, MedicationChange, PendingResult
from .pdf_loader import load_patient_folder
from .rag import RAGPipeline
from .state import AgentState
from .tools import drug_interaction_lookup, escalate_to_clinician
from .tracer import AgentTracer

logger = logging.getLogger(__name__)

ALL_SECTIONS = [
    "demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "secondary_diagnoses",
    "hospital_course",
    "procedures",
    "admission_medications",
    "discharge_medications",
    "allergies",
    "follow_up",
    "pending_results",
    "discharge_condition",
]

# Shared instances (per run, injected via closure)
_rag: RAGPipeline | None = None
_tracer: AgentTracer | None = None
_llm: ChatAnthropic | None = None


def init_run_context(rag: RAGPipeline, tracer: AgentTracer, llm: ChatAnthropic) -> None:
    global _rag, _tracer, _llm
    _rag = rag
    _tracer = tracer
    _llm = llm


def _llm_extract(section: str, context: str, instruction: str) -> str:
    """Call LLM to extract a specific field from context. Never fabricate."""
    system = (
        "You are a clinical information extractor. "
        "Extract ONLY what is explicitly stated in the provided clinical notes. "
        "NEVER invent, infer, or guess clinical facts. "
        "If information is absent, return exactly: MISSING "
        "If information is pending/not yet available, return exactly: PENDING "
        "If there is a conflict between sources, return exactly: CONFLICT: <describe the conflict> "
        "Be concise and factual."
    )
    prompt = (
        f"Section: {section}\n"
        f"Task: {instruction}\n\n"
        f"Clinical notes context:\n{context}\n\n"
        f"Extracted value (or MISSING/PENDING/CONFLICT):"
    )
    try:
        response = _llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed for {section}: {e}")
        return "MISSING"


def node_ingest_documents(state: AgentState) -> dict:
    """Load PDFs and build RAG index."""
    _tracer.start_step("ingest_documents")
    folder = Path(state["patient_folder"])

    docs = load_patient_folder(folder)
    doc_summary = {k: ("ok" if not v.error else f"error: {v.error}") for k, v in docs.items()}

    n_chunks = _rag.index_documents(docs)

    result = {
        "documents_loaded": True,
        "indexed_chunks": n_chunks,
        "raw_context": {},
        "errors": state.get("errors", []),
    }

    if n_chunks == 0:
        result["errors"] = result["errors"] + ["No content could be extracted from any document"]
        result["should_stop"] = True
        result["stop_reason"] = "No indexable content"

    _tracer.record(
        node="ingest_documents",
        reasoning=f"Loaded {len(docs)} PDF(s) from {folder.name}",
        action="load_patient_folder + rag.index_documents",
        inputs={"folder": str(folder), "docs": list(docs.keys())},
        result=f"{n_chunks} chunks indexed. Docs: {doc_summary}",
        next_decision="proceed to planning" if n_chunks > 0 else "stop — no content",
    )
    return result


def node_plan(state: AgentState) -> dict:
    """Plan which sections to extract based on available content."""
    _tracer.start_step("plan")

    # Quick probe: which sections have any relevant content?
    present = []
    absent = []
    for section in ALL_SECTIONS:
        probe = _rag.query(section, n_results=2)
        if probe:
            present.append(section)
        else:
            absent.append(section)

    pending_sections = present + absent  # try all, absent ones will be marked MISSING

    _tracer.record(
        node="plan",
        reasoning=f"RAG probe found content for {len(present)}/{len(ALL_SECTIONS)} sections",
        action="rag.query probe for each section",
        inputs={"total_sections": len(ALL_SECTIONS)},
        result=f"Sections with content: {present}\nLikely absent: {absent}",
        next_decision=f"extract {len(pending_sections)} sections sequentially",
    )

    return {
        "pending_sections": pending_sections,
        "completed_sections": [],
        "summary": DischargeSummary(),
        "drug_interaction_results": [],
        "escalation_log": [],
    }


def node_extract_section(state: AgentState) -> dict:
    """Extract the next pending section from RAG + LLM."""
    pending = list(state["pending_sections"])
    if not pending:
        return state

    section = pending[0]
    _tracer.start_step(f"extract_{section}")

    context = _rag.query_section(section)
    summary: DischargeSummary = state["summary"]
    raw_context = dict(state.get("raw_context", {}))
    raw_context[section] = context

    value = _llm_extract(section, context, _section_instruction(section))

    # Parse and populate summary fields
    _apply_extracted_value(summary, section, value)

    _tracer.record(
        node=f"extract_{section}",
        reasoning=f"Querying RAG for section '{section}', then LLM extraction",
        action="rag.query_section → llm_extract",
        inputs={"section": section, "context_chars": len(context)},
        result=f"Extracted: {value[:200]}",
        next_decision=f"{len(pending)-1} sections remaining",
    )

    return {
        "pending_sections": pending[1:],
        "completed_sections": state.get("completed_sections", []) + [section],
        "summary": summary,
        "raw_context": raw_context,
        "iteration": state["iteration"] + 1,
    }


def _section_instruction(section: str) -> str:
    instructions = {
        "demographics": "Extract patient full name, date of birth, and MRN/patient ID as a JSON object with keys: name, dob, mrn.",
        "admission_date": "Extract the hospital admission date (format: YYYY-MM-DD if possible, otherwise as written).",
        "discharge_date": "Extract the hospital discharge date (format: YYYY-MM-DD if possible, otherwise as written).",
        "principal_diagnosis": "Extract the single principal/primary diagnosis.",
        "secondary_diagnoses": "Extract all secondary diagnoses as a comma-separated list.",
        "hospital_course": "Summarize the hospital course in 2-4 sentences covering key events, treatments, and response.",
        "procedures": "List all procedures and surgical interventions performed during admission.",
        "admission_medications": "List all medications the patient was taking on admission (home medications).",
        "discharge_medications": "List all medications prescribed at discharge, with doses and frequencies.",
        "allergies": "List all documented allergies and reactions. If NKDA or no known allergies, state that.",
        "follow_up": "Extract all follow-up instructions, appointments, and referrals.",
        "pending_results": "List any lab results, cultures, or studies that are still pending/outstanding.",
        "discharge_condition": "State the patient's condition at time of discharge (e.g., stable, improved, critical).",
    }
    return instructions.get(section, f"Extract information relevant to {section}.")


def _apply_extracted_value(summary: DischargeSummary, section: str, value: str) -> None:
    """Parse extracted value and apply to summary, flagging issues."""
    is_missing = value.strip().upper().startswith("MISSING")
    is_pending = value.strip().upper().startswith("PENDING")
    is_conflict = value.strip().upper().startswith("CONFLICT")

    if is_missing:
        summary.mark_missing(section)
        return
    if is_pending:
        summary.add_flag(FlagLevel.WARNING, section, f"Data pending: {value}")
        if section == "pending_results":
            summary.pending_results.append(PendingResult(test_name="Unknown", note=value))
        return
    if is_conflict:
        summary.add_flag(FlagLevel.CRITICAL, section, f"Conflict detected: {value}")
        return

    # Apply to fields
    if section == "demographics":
        try:
            # Try JSON parse first
            clean = value.strip().strip("```json").strip("```").strip()
            data = json.loads(clean)
            summary.patient_name = data.get("name") or summary.patient_name
            summary.date_of_birth = data.get("dob") or summary.date_of_birth
            summary.mrn = data.get("mrn") or summary.mrn
        except Exception:
            # Fallback: treat as plain text
            summary.patient_name = value
    elif section == "admission_date":
        summary.admission_date = value
    elif section == "discharge_date":
        summary.discharge_date = value
    elif section == "principal_diagnosis":
        summary.principal_diagnosis = value
    elif section == "secondary_diagnoses":
        if value and value.upper() != "NONE":
            summary.secondary_diagnoses = [d.strip() for d in value.split(",") if d.strip()]
    elif section == "hospital_course":
        summary.hospital_course = value
    elif section == "procedures":
        if value and value.upper() != "NONE":
            summary.procedures = [p.strip() for p in value.split(",") if p.strip()]
    elif section == "admission_medications":
        if value and value.upper() != "NONE":
            summary.admission_medications = [m.strip() for m in value.split("\n") if m.strip()]
    elif section == "discharge_medications":
        if value and value.upper() != "NONE":
            summary.discharge_medications = [m.strip() for m in value.split("\n") if m.strip()]
    elif section == "allergies":
        summary.allergies = [a.strip() for a in value.split(",") if a.strip()]
    elif section == "follow_up":
        summary.follow_up_instructions = [f.strip() for f in value.split("\n") if f.strip()]
    elif section == "pending_results":
        if value and value.upper() not in ("NONE", "MISSING"):
            for line in value.split("\n"):
                if line.strip():
                    summary.pending_results.append(PendingResult(test_name=line.strip(), note="Pending at discharge"))
    elif section == "discharge_condition":
        summary.discharge_condition = value


def node_reconcile_medications(state: AgentState) -> dict:
    """Compare admission vs discharge medications and flag changes."""
    _tracer.start_step("reconcile_medications")
    summary: DischargeSummary = state["summary"]

    admission_meds = set(m.lower() for m in summary.admission_medications)
    discharge_meds = set(m.lower() for m in summary.discharge_medications)

    changes = []

    # New medications at discharge
    for med in summary.discharge_medications:
        med_lower = med.lower()
        if not any(med_lower in a or a in med_lower for a in admission_meds):
            change = MedicationChange(
                medication=med,
                change_type="added",
                discharge_dose=med,
                reason=None,
                flagged=True,
                flag_note="Added at discharge — no documented reason found. FLAG FOR RECONCILIATION.",
            )
            changes.append(change)
            summary.add_flag(
                FlagLevel.WARNING,
                "medications",
                f"Medication added at discharge with no documented reason: {med}",
            )

    # Stopped medications
    for med in summary.admission_medications:
        med_lower = med.lower()
        if not any(med_lower in d or d in med_lower for d in discharge_meds):
            change = MedicationChange(
                medication=med,
                change_type="stopped",
                admission_dose=med,
                reason=None,
                flagged=True,
                flag_note="Stopped at discharge — no documented reason found. FLAG FOR RECONCILIATION.",
            )
            changes.append(change)
            summary.add_flag(
                FlagLevel.WARNING,
                "medications",
                f"Medication stopped at discharge with no documented reason: {med}",
            )

    # Continued medications
    for med in summary.discharge_medications:
        med_lower = med.lower()
        if any(med_lower in a or a in med_lower for a in admission_meds):
            changes.append(MedicationChange(medication=med, change_type="continued"))

    summary.medication_changes = changes

    if not summary.admission_medications and not summary.discharge_medications:
        summary.add_flag(FlagLevel.WARNING, "medications", "No medication lists found in documents")
        summary.mark_missing("medications")

    _tracer.record(
        node="reconcile_medications",
        reasoning="Comparing admission vs discharge medication lists for changes",
        action="set difference + flag undocumented changes",
        inputs={"admission": len(summary.admission_medications), "discharge": len(summary.discharge_medications)},
        result=f"{len(changes)} changes identified ({sum(1 for c in changes if c.flagged)} flagged)",
        next_decision="proceed to conflict detection",
    )

    return {"summary": summary}


def node_check_conflicts(state: AgentState) -> dict:
    """Detect conflicts between document sources."""
    _tracer.start_step("check_conflicts")
    summary: DischargeSummary = state["summary"]
    raw_context = state.get("raw_context", {})

    conflicts_found = []

    # Re-query key sections to look for disagreement
    conflict_sections = [
        ("principal_diagnosis", "What is the discharge diagnosis?"),
        ("discharge_date", "What is the discharge date?"),
        ("discharge_condition", "What is the patient condition at discharge?"),
    ]

    for section, query in conflict_sections:
        chunks = _rag.query(query, n_results=6)
        # Look for keyword disagreement signals
        texts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]
        unique_sources = list(set(sources))

        if len(unique_sources) > 1:
            # Ask LLM to check for conflicts
            context = "\n---\n".join(texts[:4])
            conflict_check = _llm_extract(
                section,
                context,
                f"Do multiple notes DISAGREE about the {section.replace('_', ' ')}? "
                "If yes, return: CONFLICT: <describe>. If no conflict, return: NO CONFLICT.",
            )
            if "CONFLICT" in conflict_check.upper() and "NO CONFLICT" not in conflict_check.upper():
                summary.add_flag(
                    FlagLevel.CRITICAL,
                    section,
                    f"Conflicting information across documents: {conflict_check}",
                )
                conflicts_found.append(section)

    _tracer.record(
        node="check_conflicts",
        reasoning="Checking for contradictions between document sources on key fields",
        action="rag.query + llm conflict detection",
        inputs={"sections_checked": [s for s, _ in conflict_sections]},
        result=f"Conflicts found in: {conflicts_found or 'none'}",
        next_decision="proceed to drug interaction check",
    )

    return {"summary": summary}


def node_drug_interaction_check(state: AgentState) -> dict:
    """Check discharge medications for dangerous interactions."""
    _tracer.start_step("drug_interaction_check")
    summary: DischargeSummary = state["summary"]
    escalation_log = list(state.get("escalation_log", []))

    if not summary.discharge_medications:
        _tracer.record(
            node="drug_interaction_check",
            reasoning="No discharge medications to check",
            action="skipped",
            inputs={},
            result="No medications",
            next_decision="proceed to compile summary",
        )
        return {}

    result = drug_interaction_lookup(summary.discharge_medications)

    for interaction in result["interactions"]:
        severity = interaction["severity"].split(":")[0].strip()
        flag_level = FlagLevel.CRITICAL if severity == "HIGH" else FlagLevel.WARNING
        summary.add_flag(
            flag_level,
            "drug_interactions",
            f"Interaction: {interaction['drug1']} + {interaction['drug2']} → {interaction['severity']}",
        )
        # Escalate HIGH severity interactions
        if severity == "HIGH":
            esc = escalate_to_clinician(
                patient_id=state["patient_id"],
                severity="HIGH",
                section="drug_interactions",
                message=f"Drug interaction: {interaction['drug1']} + {interaction['drug2']} — {interaction['severity']}",
            )
            escalation_log.append(esc)

    _tracer.record(
        node="drug_interaction_check",
        reasoning=f"Checking {len(summary.discharge_medications)} discharge medications for interactions",
        action="drug_interaction_lookup tool",
        inputs={"medications": summary.discharge_medications},
        result=f"Status: {result['status']}. Interactions: {len(result['interactions'])}. Errors: {result['errors']}",
        next_decision=f"{'escalating HIGH interactions; ' if any(i['severity'].startswith('HIGH') for i in result['interactions']) else ''}proceed to compile",
    )

    return {
        "summary": summary,
        "drug_interaction_results": result["interactions"],
        "escalation_log": escalation_log,
    }


def node_compile_summary(state: AgentState) -> dict:
    """Final validation: ensure no fabrication, mark all missing fields."""
    _tracer.start_step("compile_summary")
    summary: DischargeSummary = state["summary"]
    escalation_log = list(state.get("escalation_log", []))

    # Required fields check
    required = {
        "patient_name": summary.patient_name,
        "admission_date": summary.admission_date,
        "discharge_date": summary.discharge_date,
        "principal_diagnosis": summary.principal_diagnosis,
        "hospital_course": summary.hospital_course,
        "discharge_condition": summary.discharge_condition,
    }
    for field_name, value in required.items():
        if not value or value.upper() in ("MISSING", "NONE", "N/A"):
            summary.mark_missing(field_name)
            summary.add_flag(
                FlagLevel.WARNING,
                field_name,
                f"Required field '{field_name}' could not be sourced from documents. CLINICIAN REVIEW REQUIRED.",
            )

    # Escalate critical flags
    critical_flags = [f for f in summary.flags if f.level == FlagLevel.CRITICAL]
    for flag in critical_flags:
        esc = escalate_to_clinician(
            patient_id=state["patient_id"],
            severity="CRITICAL",
            section=flag.section,
            message=flag.message,
        )
        escalation_log.append(esc)

    _tracer.record(
        node="compile_summary",
        reasoning="Final validation: check required fields, escalate critical issues",
        action="field validation + escalate_to_clinician for CRITICAL flags",
        inputs={"required_fields": list(required.keys())},
        result=(
            f"Missing: {summary.missing_fields}\n"
            f"Total flags: {len(summary.flags)} "
            f"({sum(1 for f in summary.flags if f.level == FlagLevel.CRITICAL)} critical)\n"
            f"Escalations: {len(escalation_log)}"
        ),
        next_decision="agent complete",
    )

    return {
        "summary": summary,
        "escalation_log": escalation_log,
        "should_stop": True,
        "stop_reason": "Summary complete",
    }
