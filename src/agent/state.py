"""LangGraph agent state."""
from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict
from src.agent.models import DischargeSummary


class AgentState(TypedDict):
    # Input
    patient_id:     str
    patient_folder: str

    # Runtime control
    iteration:      int
    max_iterations: int
    should_stop:    bool
    stop_reason:    str

    # Document context
    documents_loaded: bool
    indexed_chunks:   int
    raw_context:      dict[str, str]

    # Section processing queue
    pending_sections:   list[str]
    completed_sections: list[str]

    # Output being built
    summary: Optional[DischargeSummary]

    # Tool results
    drug_interaction_results: list[dict[str, Any]]
    escalation_log:           list[dict[str, Any]]

    # Diagnostics
    errors:      list[str]
    last_node:   str
    last_action: str
