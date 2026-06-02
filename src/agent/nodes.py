"""LangGraph node implementations — all nodes use src.logger and src.exception."""
from __future__ import annotations
import json
import logging
import sys
import concurrent.futures
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.models import (
    DischargeSummary, FlagLevel, MedicationChange, PendingResult,
)
from src.agent.state import AgentState
from src.agent.tracer import AgentTracer
from src.components.agent_tools import DrugInteractionTool, EscalationTool
from src.components.pdf_loader import PDFLoader
from src.components.rag_pipeline import RAGPipeline
from src.constants import ALL_SECTIONS, SECTION_INSTRUCTIONS, SECTION_QUERIES
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)

# ── Shared run-context ────────────────────────────────────────────────────────
_rag:        RAGPipeline | None = None
_tracer:     AgentTracer | None = None
_llm:        ChatAnthropic | None = None
_drug_tool:  DrugInteractionTool | None = None
_esc_tool:   EscalationTool | None = None
_correction_memory = None   # Optional[CorrectionMemory] — injected for Part 2


def init_run_context(
    rag: RAGPipeline,
    tracer: AgentTracer,
    llm: ChatAnthropic,
    correction_memory=None,
) -> None:
    global _rag, _tracer, _llm, _drug_tool, _esc_tool, _correction_memory
    _rag               = rag
    _tracer            = tracer
    _llm               = llm
    _drug_tool         = DrugInteractionTool()
    _esc_tool          = EscalationTool()
    _correction_memory = correction_memory


# ── Node 1: Ingest documents ─────────────────────────────────────────────────

def node_ingest_documents(state: AgentState) -> dict:
    """
    Load PDFs and index into RAG.
    If RAG is already indexed (pre-loaded from Stage 2), skip re-loading entirely.
    """
    try:
        _tracer.start_step("ingest_documents")

        # Fast path: RAG already has indexed chunks (passed from Stage 2)
        if _rag._store is not None:
            existing = getattr(_rag._store, '_collection', None)
            count = existing.count() if existing else (
                len(getattr(_rag._store, '_docs', [])))
            if count > 0:
                _tracer.record(
                    node="ingest_documents",
                    reasoning="RAG already indexed from Stage 2 — skipping re-load",
                    action="skip (pre-indexed)",
                    inputs={"chunks": count},
                    result=f"{count} chunks already available",
                    next_decision="proceed to batch extraction",
                )
                return {
                    "documents_loaded": True,
                    "indexed_chunks": count,
                    "raw_context": {},
                    "errors": state.get("errors", []),
                }

        # Slow path: load and index from disk
        folder = Path(state["patient_folder"])
        loader = PDFLoader()
        docs   = loader.load_folder(folder)
        doc_status = {k: ("ok" if not v.error else f"err: {v.error}") for k, v in docs.items()}

        n_chunks = _rag.index_documents(docs)
        result: dict[str, Any] = {
            "documents_loaded": True,
            "indexed_chunks":   n_chunks,
            "raw_context":      {},
            "errors":           state.get("errors", []),
        }
        if n_chunks == 0:
            result["errors"]      = result["errors"] + ["No extractable content"]
            result["should_stop"] = True
            result["stop_reason"] = "No indexable content"

        _tracer.record(
            node="ingest_documents",
            reasoning=f"Loading {len(docs)} PDF(s) from {folder.name}",
            action="PDFLoader.load_folder + RAGPipeline.index_documents",
            inputs={"folder": str(folder), "docs": list(docs.keys())},
            result=f"{n_chunks} chunks indexed. {doc_status}",
            next_decision="proceed to batch extraction" if n_chunks > 0 else "stop",
        )
        return result
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Node 2: Plan ─────────────────────────────────────────────────────────────

def node_plan(state: AgentState) -> dict:
    """Plan: probe RAG for available sections."""
    try:
        _tracer.start_step("plan")

        # Parallel RAG probes for all sections
        def probe(section):
            hits = _rag.query(section, n_results=1)
            return section, bool(hits)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            probe_results = list(ex.map(lambda s: probe(s), ALL_SECTIONS))

        present = [s for s, found in probe_results if found]
        absent  = [s for s, found in probe_results if not found]

        _tracer.record(
            node="plan",
            reasoning=f"Parallel RAG probe: {len(present)}/{len(ALL_SECTIONS)} sections have content",
            action="parallel rag.query probe",
            inputs={"sections": len(ALL_SECTIONS)},
            result=f"Present: {present}\nAbsent: {absent}",
            next_decision="batch LLM extraction",
        )
        return {
            "pending_sections":   ALL_SECTIONS,   # batch node handles all at once
            "completed_sections": [],
            "summary":            DischargeSummary(),
            "drug_interaction_results": [],
            "escalation_log":     [],
        }
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Node 3: Batch extraction (replaces 13 sequential LLM calls) ──────────────

def node_batch_extract(state: AgentState) -> dict:
    """
    Extract ALL sections in a SINGLE LLM call.
    Collects RAG context for all sections in parallel, then sends one prompt.
    13 sequential calls → 1 batched call.
    """
    try:
        _tracer.start_step("batch_extract")

        # Step A: gather context for all sections in parallel
        def fetch_context(section):
            return section, _rag.query_section(section)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            ctx_results = dict(ex.map(lambda s: fetch_context(s), ALL_SECTIONS))

        # Step B: build one combined prompt — inject correction memory if available
        memory_block = ""
        if _correction_memory is not None:
            memory_block = _correction_memory.get_prompt_injection(ALL_SECTIONS)
            if memory_block:
                logger.info("Injecting correction memory into extraction prompt")

        system_prompt = (
            "You are a clinical information extractor.\n"
            "Extract ONLY what is explicitly stated in the provided clinical notes.\n"
            "NEVER invent, infer, or guess clinical facts.\n"
            "Rules:\n"
            "  - If a field is absent → value: \"MISSING\"\n"
            "  - If a result is still pending → value: \"PENDING\"\n"
            "  - If two notes contradict each other → value: \"CONFLICT: <describe>\"\n"
            "Return a single JSON object with exactly these keys:\n"
            "demographics, admission_date, discharge_date, principal_diagnosis,\n"
            "secondary_diagnoses, hospital_course, procedures, admission_medications,\n"
            "discharge_medications, allergies, follow_up, pending_results, discharge_condition\n\n"
            "For demographics return a nested object: {name, dob, mrn}\n"
            "For list fields (medications, procedures, etc.) return newline-separated strings.\n"
            f"{memory_block}"
            "Return ONLY the JSON — no explanation, no markdown fences."
        )

        sections_text = "\n\n".join(
            f"=== {section.upper()} ===\n"
            f"Task: {SECTION_INSTRUCTIONS.get(section, section)}\n"
            f"Context:\n{ctx_results[section]}"
            for section in ALL_SECTIONS
        )
        user_prompt = (
            f"Extract all discharge summary fields from the following clinical note contexts.\n\n"
            f"{sections_text}\n\n"
            f"Return JSON only:"
        )

        # Step C: single LLM call
        logger.info("Batch extraction: 1 LLM call for %d sections", len(ALL_SECTIONS))
        try:
            response = _llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw = response.content.strip().strip("```json").strip("```").strip()
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            # If JSON parse fails, fall back to per-section extraction
            logger.warning("Batch JSON parse failed — falling back to per-section extraction")
            extracted = _fallback_per_section(ctx_results)
        except Exception as llm_err:
            logger.error("Batch LLM call failed: %s", llm_err)
            extracted = {}

        # Step D: apply extracted values to summary
        summary: DischargeSummary = state["summary"]
        for section in ALL_SECTIONS:
            value = extracted.get(section, "MISSING")
            if isinstance(value, dict):          # demographics sub-object
                value = json.dumps(value)
            _apply_value(summary, section, str(value))

        _tracer.record(
            node="batch_extract",
            reasoning=f"Parallel context fetch + single LLM call for all {len(ALL_SECTIONS)} sections",
            action="parallel rag.query_section → 1x llm.invoke → parse JSON",
            inputs={"sections": len(ALL_SECTIONS)},
            result=f"Extracted {len(extracted)} sections. Missing: {summary.missing_fields}",
            next_decision="reconcile medications",
        )
        return {
            "pending_sections":   [],
            "completed_sections": ALL_SECTIONS,
            "summary":            summary,
            "raw_context":        ctx_results,
            "iteration":          state["iteration"] + 1,
        }
    except Exception as e:
        raise DischargeAgentException(e, sys)


def _fallback_per_section(ctx_results: dict) -> dict:
    """Per-section LLM extraction used when batch JSON parsing fails."""
    results = {}

    def extract_one(section):
        context     = ctx_results.get(section, "")
        instruction = SECTION_INSTRUCTIONS.get(section, f"Extract {section}.")
        system = (
            "Extract ONLY what is stated. Return MISSING/PENDING/CONFLICT: as needed. "
            "Be concise."
        )
        prompt = f"Section: {section}\nTask: {instruction}\n\nContext:\n{context}\n\nValue:"
        try:
            r = _llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            return section, r.content.strip()
        except Exception:
            return section, "MISSING"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        for section, value in ex.map(extract_one, ALL_SECTIONS):
            results[section] = value

    return results


# ── Node 4: Reconcile medications ────────────────────────────────────────────

def node_reconcile_medications(state: AgentState) -> dict:
    """Compare admission vs discharge meds, flag undocumented changes."""
    try:
        _tracer.start_step("reconcile_medications")
        summary: DischargeSummary = state["summary"]
        admission = set(m.lower() for m in summary.admission_medications)
        discharge = set(m.lower() for m in summary.discharge_medications)
        changes   = []

        for med in summary.discharge_medications:
            ml = med.lower()
            if not any(ml in a or a in ml for a in admission):
                changes.append(MedicationChange(
                    medication=med, change_type="added", discharge_dose=med,
                    flagged=True,
                    flag_note="Added at discharge — no documented reason. FLAG FOR RECONCILIATION.",
                ))
                summary.add_flag(FlagLevel.WARNING, "medications",
                    f"Med added at discharge with no documented reason: {med}")

        for med in summary.admission_medications:
            ml = med.lower()
            if not any(ml in d or d in ml for d in discharge):
                changes.append(MedicationChange(
                    medication=med, change_type="stopped", admission_dose=med,
                    flagged=True,
                    flag_note="Stopped at discharge — no documented reason. FLAG FOR RECONCILIATION.",
                ))
                summary.add_flag(FlagLevel.WARNING, "medications",
                    f"Med stopped at discharge with no documented reason: {med}")

        for med in summary.discharge_medications:
            ml = med.lower()
            if any(ml in a or a in ml for a in admission):
                changes.append(MedicationChange(medication=med, change_type="continued"))

        if not summary.admission_medications and not summary.discharge_medications:
            summary.add_flag(FlagLevel.WARNING, "medications", "No medication lists found")
            summary.mark_missing("medications")

        summary.medication_changes = changes
        n_flagged = sum(1 for c in changes if c.flagged)

        _tracer.record(
            node="reconcile_medications",
            reasoning="Comparing admission vs discharge medication lists",
            action="set-difference + flag undocumented changes",
            inputs={"admission": len(summary.admission_medications),
                    "discharge": len(summary.discharge_medications)},
            result=f"{len(changes)} changes ({n_flagged} flagged)",
            next_decision="parallel: conflict check + drug interaction check",
        )
        return {"summary": summary}
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Node 5: Check conflicts ───────────────────────────────────────────────────

def node_check_conflicts(state: AgentState) -> dict:
    """Detect contradictions between document sources."""
    try:
        _tracer.start_step("check_conflicts")
        summary: DischargeSummary = state["summary"]
        conflicts_found = []

        conflict_checks = [
            ("principal_diagnosis", "What is the discharge diagnosis?"),
            ("discharge_date",      "What is the discharge date?"),
            ("discharge_condition", "What is the patient condition at discharge?"),
        ]

        def check_one(args):
            section, query = args
            chunks = _rag.query(query, n_results=6)
            if len(set(c["source"] for c in chunks)) <= 1:
                return section, None
            context = "\n---\n".join(c["text"] for c in chunks[:4])
            try:
                r = _llm.invoke([
                    SystemMessage(content="Detect conflicts between clinical notes. Be concise."),
                    HumanMessage(content=(
                        f"Do multiple notes DISAGREE about the {section.replace('_', ' ')}?\n"
                        f"Context:\n{context}\n\n"
                        "Return CONFLICT: <describe> or NO CONFLICT."
                    )),
                ])
                return section, r.content.strip()
            except Exception:
                return section, None

        # Run all conflict checks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(check_one, conflict_checks))

        for section, check in results:
            if check and "CONFLICT" in check.upper() and "NO CONFLICT" not in check.upper():
                summary.add_flag(FlagLevel.CRITICAL, section,
                    f"Conflicting information: {check}")
                conflicts_found.append(section)

        _tracer.record(
            node="check_conflicts",
            reasoning="Parallel conflict detection across document sources",
            action="parallel rag.query + llm conflict check",
            inputs={"sections_checked": [s for s, _ in conflict_checks]},
            result=f"Conflicts in: {conflicts_found or 'none'}",
            next_decision="drug interaction check",
        )
        return {"summary": summary}
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Node 6: Drug interaction check ───────────────────────────────────────────

def node_drug_interaction_check(state: AgentState) -> dict:
    """Check discharge medications for dangerous interactions."""
    try:
        _tracer.start_step("drug_interaction_check")
        summary: DischargeSummary = state["summary"]
        escalation_log = list(state.get("escalation_log", []))

        if not summary.discharge_medications:
            _tracer.record(
                node="drug_interaction_check", reasoning="No discharge medications",
                action="skipped", inputs={}, result="No medications",
                next_decision="compile summary",
            )
            return {}

        result = _drug_tool.check(summary.discharge_medications)

        for interaction in result["interactions"]:
            severity   = interaction["severity"].split(":")[0].strip()
            flag_level = FlagLevel.CRITICAL if severity == "HIGH" else FlagLevel.WARNING
            summary.add_flag(flag_level, "drug_interactions",
                f"{interaction['drug1']} + {interaction['drug2']} → {interaction['severity']}")
            if severity == "HIGH":
                esc = _esc_tool.escalate(
                    patient_id=state["patient_id"], severity="HIGH",
                    section="drug_interactions",
                    message=f"{interaction['drug1']} + {interaction['drug2']}: {interaction['severity']}",
                )
                escalation_log.append(esc)

        _tracer.record(
            node="drug_interaction_check",
            reasoning=f"Checking {len(summary.discharge_medications)} medications",
            action="DrugInteractionTool.check",
            inputs={"medications": summary.discharge_medications},
            result=f"{len(result['interactions'])} interaction(s) | status: {result['status']}",
            next_decision="compile summary",
        )
        return {
            "summary":                  summary,
            "drug_interaction_results": result["interactions"],
            "escalation_log":           escalation_log,
        }
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Node 7: Compile summary ───────────────────────────────────────────────────

def node_compile_summary(state: AgentState) -> dict:
    """Final validation: check required fields, escalate CRITICAL flags."""
    try:
        _tracer.start_step("compile_summary")
        summary: DischargeSummary = state["summary"]
        escalation_log = list(state.get("escalation_log", []))

        required = {
            "patient_name":       summary.patient_name,
            "admission_date":     summary.admission_date,
            "discharge_date":     summary.discharge_date,
            "principal_diagnosis":summary.principal_diagnosis,
            "hospital_course":    summary.hospital_course,
            "discharge_condition":summary.discharge_condition,
        }
        for field_name, value in required.items():
            if not value or str(value).upper() in ("MISSING", "NONE", "N/A"):
                summary.mark_missing(field_name)
                summary.add_flag(FlagLevel.WARNING, field_name,
                    f"Required field '{field_name}' not found — CLINICIAN REVIEW REQUIRED.")

        for flag in summary.flags:
            if flag.level == FlagLevel.CRITICAL:
                esc = _esc_tool.escalate(
                    patient_id=state["patient_id"], severity="CRITICAL",
                    section=flag.section, message=flag.message,
                )
                escalation_log.append(esc)

        n_critical = sum(1 for f in summary.flags if f.level == FlagLevel.CRITICAL)
        _tracer.record(
            node="compile_summary",
            reasoning="Validate required fields + escalate CRITICAL flags",
            action="field validation + EscalationTool.escalate",
            inputs={"required_fields": list(required.keys())},
            result=(f"Missing: {summary.missing_fields} | "
                    f"Flags: {len(summary.flags)} ({n_critical} critical) | "
                    f"Escalations: {len(escalation_log)}"),
            next_decision="done",
        )
        return {
            "summary":        summary,
            "escalation_log": escalation_log,
            "should_stop":    True,
            "stop_reason":    "Summary complete",
        }
    except Exception as e:
        raise DischargeAgentException(e, sys)


# ── Value parser (shared) ─────────────────────────────────────────────────────

def _apply_value(summary: DischargeSummary, section: str, value: str) -> None:
    v = value.strip()
    if not v or v.upper() == "MISSING":
        summary.mark_missing(section)
        return
    if v.upper().startswith("PENDING"):
        summary.add_flag(FlagLevel.WARNING, section, f"Data pending: {v}")
        return
    if v.upper().startswith("CONFLICT"):
        summary.add_flag(FlagLevel.CRITICAL, section, f"Conflict: {v}")
        summary.mark_missing(section)
        return

    if section == "demographics":
        try:
            clean = v.strip("```json").strip("```").strip()
            data  = json.loads(clean)
            summary.patient_name  = data.get("name") or summary.patient_name
            summary.date_of_birth = data.get("dob")  or summary.date_of_birth
            summary.mrn           = data.get("mrn")  or summary.mrn
        except Exception:
            summary.patient_name = v
    elif section == "admission_date":     summary.admission_date = v
    elif section == "discharge_date":     summary.discharge_date = v
    elif section == "principal_diagnosis":summary.principal_diagnosis = v
    elif section == "secondary_diagnoses":
        if v.upper() not in ("NONE", "MISSING"):
            summary.secondary_diagnoses = [d.strip() for d in v.split(",") if d.strip()]
    elif section == "hospital_course":    summary.hospital_course = v
    elif section == "procedures":
        if v.upper() not in ("NONE", "MISSING"):
            summary.procedures = [p.strip() for p in v.split(",") if p.strip()]
    elif section == "admission_medications":
        if v.upper() not in ("NONE", "MISSING"):
            summary.admission_medications = [m.strip() for m in v.split("\n") if m.strip()]
    elif section == "discharge_medications":
        if v.upper() not in ("NONE", "MISSING"):
            summary.discharge_medications = [m.strip() for m in v.split("\n") if m.strip()]
    elif section == "allergies":
        summary.allergies = [a.strip() for a in v.split(",") if a.strip()]
    elif section == "follow_up":
        summary.follow_up_instructions = [f.strip() for f in v.split("\n") if f.strip()]
    elif section == "pending_results":
        if v.upper() not in ("NONE", "MISSING"):
            for line in v.split("\n"):
                if line.strip():
                    summary.pending_results.append(
                        PendingResult(test_name=line.strip(), note="Pending at discharge")
                    )
    elif section == "discharge_condition":
        summary.discharge_condition = v
