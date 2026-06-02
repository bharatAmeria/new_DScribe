"""LangGraph graph — optimised: batch LLM extraction + parallel post-processing."""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    init_run_context,
    node_batch_extract,
    node_check_conflicts,
    node_compile_summary,
    node_drug_interaction_check,
    node_ingest_documents,
    node_plan,
    node_reconcile_medications,
)
from src.agent.state import AgentState
from src.agent.tracer import AgentTracer
from src.components.rag_pipeline import RAGPipeline
from src.config import CONFIG
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


# ── Edge conditions ───────────────────────────────────────────────────────────

def _after_ingest(state: AgentState) -> str:
    return "stop" if state.get("should_stop") else "continue"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ingest_documents",       node_ingest_documents)
    graph.add_node("plan",                   node_plan)
    graph.add_node("batch_extract",          node_batch_extract)        # ← replaces 13-node loop
    graph.add_node("reconcile_medications",  node_reconcile_medications)
    graph.add_node("check_conflicts",        node_check_conflicts)
    graph.add_node("drug_interaction_check", node_drug_interaction_check)
    graph.add_node("compile_summary",        node_compile_summary)

    graph.add_edge(START, "ingest_documents")
    graph.add_conditional_edges(
        "ingest_documents",
        _after_ingest,
        {"continue": "plan", "stop": END},
    )
    graph.add_edge("plan",                   "batch_extract")
    graph.add_edge("batch_extract",          "reconcile_medications")
    graph.add_edge("reconcile_medications",  "check_conflicts")
    graph.add_edge("check_conflicts",        "drug_interaction_check")
    graph.add_edge("drug_interaction_check", "compile_summary")
    graph.add_edge("compile_summary",        END)

    return graph.compile()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent(
    patient_id: str,
    patient_folder: str | Path,
    output_dir: str | Path,
    max_iterations: int | None = None,
    pre_indexed_rag: Optional[RAGPipeline] = None,
    correction_memory=None,   # Optional[CorrectionMemory] — injected by learning loop
) -> dict:
    """
    Run the discharge summary agent for a single patient.

    Pass pre_indexed_rag (from Stage 2) to skip re-loading and re-OCR entirely.
    Returns: {summary, trace_path, escalation_log, errors, stop_reason}
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if max_iterations is None:
            max_iterations = CONFIG.agent.max_iterations

        rag    = pre_indexed_rag or RAGPipeline(patient_id=patient_id)
        tracer = AgentTracer(patient_id=patient_id, output_dir=output_dir)
        llm    = ChatAnthropic(
            model=os.getenv("DISCHARGE_MODEL", CONFIG.agent.model),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=CONFIG.agent.max_tokens,
        )
        init_run_context(rag=rag, tracer=tracer, llm=llm,
                         correction_memory=correction_memory)

        compiled = build_graph()
        initial_state: AgentState = {
            "patient_id":               patient_id,
            "patient_folder":           str(patient_folder),
            "iteration":                0,
            "max_iterations":           max_iterations,
            "should_stop":              False,
            "stop_reason":              "",
            "documents_loaded":         pre_indexed_rag is not None,
            "indexed_chunks":           0,
            "raw_context":              {},
            "pending_sections":         [],
            "completed_sections":       [],
            "summary":                  None,
            "drug_interaction_results": [],
            "escalation_log":           [],
            "errors":                   [],
            "last_node":                "",
            "last_action":              "",
        }

        logger.info("Starting agent for patient '%s' (pre-indexed: %s)",
                    patient_id, pre_indexed_rag is not None)
        final_state = compiled.invoke(initial_state)

        trace_path = tracer.save(f"trace_{patient_id}.json")
        summary    = final_state.get("summary")

        if summary:
            summary_json = output_dir / f"summary_{patient_id}.json"
            summary_md   = output_dir / f"summary_{patient_id}.md"
            with open(summary_json, "w") as f:
                json.dump(summary.model_dump(), f, indent=2)
            _save_markdown(summary, summary_md, patient_id)
            logger.info("Summary saved → %s", summary_json)

        return {
            "summary":        summary,
            "trace_path":     str(trace_path),
            "escalation_log": final_state.get("escalation_log", []),
            "errors":         final_state.get("errors", []),
            "stop_reason":    final_state.get("stop_reason", ""),
        }
    except Exception as e:
        raise DischargeAgentException(e, sys)


def _save_markdown(summary, path: Path, patient_id: str) -> None:
    lines = [
        f"# Discharge Summary — {patient_id}",
        f"> ⚠️ **{summary.generation_note}**",
        "",
        "## Patient Information",
        f"- **Name:** {summary.patient_name or '[MISSING]'}",
        f"- **DOB:** {summary.date_of_birth or '[MISSING]'}",
        f"- **MRN:** {summary.mrn or '[MISSING]'}",
        f"- **Admission:** {summary.admission_date or '[MISSING]'}",
        f"- **Discharge:** {summary.discharge_date or '[MISSING]'}",
        "",
        "## Diagnoses",
        f"- **Principal:** {summary.principal_diagnosis or '[MISSING]'}",
        "- **Secondary:** " + (", ".join(summary.secondary_diagnoses) or "[MISSING]"),
        "",
        "## Hospital Course",
        summary.hospital_course or "[MISSING — clinician review required]",
        "",
        "## Procedures",
        *([f"- {p}" for p in summary.procedures] or ["- None documented"]),
        "",
        "## Allergies",
        *([f"- {a}" for a in summary.allergies] or ["- [MISSING]"]),
        "",
        "## Discharge Medications",
        *([f"- {m}" for m in summary.discharge_medications] or ["- [MISSING]"]),
        "",
        "## Medication Changes from Admission",
    ]
    if summary.medication_changes:
        for c in summary.medication_changes:
            flag = f" ⚠️ {c.flag_note}" if c.flagged else ""
            lines.append(f"- [{c.change_type.upper()}] {c.medication}{flag}")
    else:
        lines.append("- No changes documented")

    lines += [
        "", "## Follow-Up Instructions",
        *([f"- {f}" for f in summary.follow_up_instructions] or ["- [MISSING]"]),
        "", "## Pending Results",
        *([f"- {r.test_name}: {r.note}" for r in summary.pending_results] or ["- None"]),
        "", "## Discharge Condition",
        summary.discharge_condition or "[MISSING]",
    ]
    if summary.missing_fields:
        lines += ["", "## ⚠️ Missing Fields (Clinician Action Required)"]
        lines += [f"- {f}" for f in summary.missing_fields]
    if summary.flags:
        lines += ["", "## 🚨 Flags & Escalations"]
        for flag in summary.flags:
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "ℹ️"}.get(flag.level, "•")
            lines.append(f"- {icon} **[{flag.level}] {flag.section}:** {flag.message}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
