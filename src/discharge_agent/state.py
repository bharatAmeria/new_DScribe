"""LangGraph agent state definition."""
from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict
from .models import DischargeSummary


class AgentState(TypedDict):
    # Input
    patient_id: str
    patient_folder: str

    # Runtime
    iteration: int
    max_iterations: int
    documents_loaded: bool
    indexed_chunks: int

    # Extracted raw context (per section)
    raw_context: dict[str, str]

    # Plan: list of sections still to process
    pending_sections: list[str]
    completed_sections: list[str]

    # Structured output being built
    summary: Optional[DischargeSummary]

    # Tool results
    drug_interaction_results: list[dict[str, Any]]
    escalation_log: list[dict[str, Any]]

    # Control
    errors: list[str]
    should_stop: bool
    stop_reason: str

    # Trace (passed to tracer externally, stored here for graph edges)
    last_node: str
    last_action: str
