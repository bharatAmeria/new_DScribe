"""LangGraph graph definition for the discharge summary agent."""
from __future__ import annotations
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from .nodes import (
    init_run_context,
    node_check_conflicts,
    node_compile_summary,
    node_drug_interaction_check,
    node_extract_section,
    node_ingest_documents,
    node_plan,
    node_reconcile_medications,
)
from .rag import RAGPipeline
from .state import AgentState
from .tracer import AgentTracer


def _check_continue_extraction(state: AgentState) -> str:
    """Route: keep extracting sections, or move to reconciliation."""
    if state.get("should_stop"):
        return "stop"
    if state["iteration"] >= state["max_iterations"]:
        return "stop_max_iterations"
    if state.get("pending_sections"):
        return "extract_more"
    return "done_extracting"


def _check_ingestion(state: AgentState) -> str:
    if state.get("should_stop"):
        return "stop"
    return "continue"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("ingest_documents", node_ingest_documents)
    graph.add_node("plan", node_plan)
    graph.add_node("extract_section", node_extract_section)
    graph.add_node("reconcile_medications", node_reconcile_medications)
    graph.add_node("check_conflicts", node_check_conflicts)
    graph.add_node("drug_interaction_check", node_drug_interaction_check)
    graph.add_node("compile_summary", node_compile_summary)

    # Edges
    graph.add_edge(START, "ingest_documents")
    graph.add_conditional_edges(
        "ingest_documents",
        _check_ingestion,
        {"continue": "plan", "stop": END},
    )
    graph.add_edge("plan", "extract_section")
    graph.add_conditional_edges(
        "extract_section",
        _check_continue_extraction,
        {
            "extract_more": "extract_section",
            "done_extracting": "reconcile_medications",
            "stop": END,
            "stop_max_iterations": "compile_summary",  # partial result if cap hit
        },
    )
    graph.add_edge("reconcile_medications", "check_conflicts")
    graph.add_edge("check_conflicts", "drug_interaction_check")
    graph.add_edge("drug_interaction_check", "compile_summary")
    graph.add_edge("compile_summary", END)

    return graph.compile()


def run_agent(
    patient_id: str,
    patient_folder: str | Path,
    output_dir: str | Path = "outputs",
    max_iterations: int = 20,
) -> dict:
    """
    Run the discharge summary agent for a single patient.
    Returns dict with summary, trace_path, escalation_log.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init shared context
    rag = RAGPipeline(collection_name=f"patient_{patient_id}")
    tracer = AgentTracer(patient_id=patient_id, output_dir=output_dir)
    llm = ChatAnthropic(
        model=os.getenv("DISCHARGE_MODEL", "claude-haiku-4-5-20251001"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )
    init_run_context(rag=rag, tracer=tracer, llm=llm)

    # Build and run graph
    compiled = build_graph()
    initial_state: AgentState = {
        "patient_id": patient_id,
        "patient_folder": str(patient_folder),
        "iteration": 0,
        "max_iterations": max_iterations,
        "documents_loaded": False,
        "indexed_chunks": 0,
        "raw_context": {},
        "pending_sections": [],
        "completed_sections": [],
        "summary": None,
        "drug_interaction_results": [],
        "escalation_log": [],
        "errors": [],
        "should_stop": False,
        "stop_reason": "",
        "last_node": "",
        "last_action": "",
    }

    final_state = compiled.invoke(initial_state)

    # Save outputs
    trace_path = tracer.save(f"trace_{patient_id}.json")
    summary = final_state.get("summary")

    if summary:
        import json
        summary_path = output_dir / f"summary_{patient_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        _save_readable_summary(summary, output_dir / f"summary_{patient_id}.md", patient_id)

    return {
        "summary": summary,
        "trace_path": str(trace_path),
        "escalation_log": final_state.get("escalation_log", []),
        "errors": final_state.get("errors", []),
        "stop_reason": final_state.get("stop_reason", ""),
    }


def _save_readable_summary(summary, path: Path, patient_id: str) -> None:
    """Save a human-readable markdown discharge summary."""
    from .models import FlagLevel

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
    ]
    if summary.secondary_diagnoses:
        lines.append("- **Secondary:**")
        for d in summary.secondary_diagnoses:
            lines.append(f"  - {d}")
    else:
        lines.append("- **Secondary:** [MISSING]")

    lines += [
        "",
        "## Hospital Course",
        summary.hospital_course or "[MISSING — clinician review required]",
        "",
        "## Procedures",
    ]
    if summary.procedures:
        for p in summary.procedures:
            lines.append(f"- {p}")
    else:
        lines.append("- None documented")

    lines += ["", "## Allergies"]
    for a in summary.allergies or ["[MISSING]"]:
        lines.append(f"- {a}")

    lines += ["", "## Medications"]
    lines.append("### Discharge Medications")
    for m in summary.discharge_medications or ["[MISSING]"]:
        lines.append(f"- {m}")

    lines.append("\n### Medication Changes from Admission")
    if summary.medication_changes:
        for c in summary.medication_changes:
            flag_note = f" ⚠️ {c.flag_note}" if c.flagged else ""
            lines.append(f"- [{c.change_type.upper()}] {c.medication}{flag_note}")
    else:
        lines.append("- No changes documented")

    lines += ["", "## Follow-Up Instructions"]
    for f in summary.follow_up_instructions or ["[MISSING]"]:
        lines.append(f"- {f}")

    lines += ["", "## Pending Results"]
    if summary.pending_results:
        for r in summary.pending_results:
            lines.append(f"- {r.test_name}: {r.note}")
    else:
        lines.append("- None documented")

    lines += [
        "",
        "## Discharge Condition",
        summary.discharge_condition or "[MISSING]",
    ]

    if summary.missing_fields:
        lines += ["", "## ⚠️ Missing Fields (Clinician Action Required)"]
        for f in summary.missing_fields:
            lines.append(f"- {f}")

    if summary.flags:
        lines += ["", "## 🚨 Flags & Escalations"]
        for flag in summary.flags:
            icon = "🔴" if flag.level == FlagLevel.CRITICAL else "🟡" if flag.level == FlagLevel.WARNING else "ℹ️"
            lines.append(f"- {icon} **[{flag.level}] {flag.section}:** {flag.message}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
