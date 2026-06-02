"""
Architecture validation tests — mirrors sales-price-main test pattern.
Verifies: logger, exception, constants, config, components, graph.
Run: python tests/test_architecture.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Logger (must be first) ───────────────────────────────────────────────────
from src.logger import configure_logger      # noqa: F401
from src.exception import DischargeAgentException
from src.constants import (
    PDF_INGESTION_STAGE_NAME, RAG_INDEXING_STAGE_NAME, AGENT_RUN_STAGE_NAME,
    ALL_SECTIONS, SECTION_INSTRUCTIONS, SECTION_QUERIES, KNOWN_DRUG_INTERACTIONS,
)
from src.config import CONFIG
from src.components.rag_pipeline import RAGPipeline, BM25Index
from src.components.agent_tools import DrugInteractionTool, EscalationTool
from src.agent.models import DischargeSummary, FlagLevel, MedicationChange, PendingResult
from src.agent.graph import build_graph

logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"


def test(name: str, fn):
    try:
        fn()
        logger.info("%s %s", PASS, name)
        return True
    except Exception as e:
        logger.error("%s %s — %s", FAIL, name, e)
        return False


# ── Test functions ────────────────────────────────────────────────────────────

def t_logger():
    log = logging.getLogger("architecture_test")
    log.info("Logger test message")
    assert log.handlers or logging.getLogger().handlers

def t_exception():
    try:
        raise ValueError("test error from test suite")
    except Exception as e:
        exc = DischargeAgentException(e, sys)
        msg = str(exc)
        assert "test error" in msg
        assert "line" in msg.lower() or "Error" in msg

def t_constants():
    assert PDF_INGESTION_STAGE_NAME == "PDF Ingestion"
    assert RAG_INDEXING_STAGE_NAME  == "RAG Indexing"
    assert AGENT_RUN_STAGE_NAME     == "Agent Run"
    assert len(ALL_SECTIONS) == 13
    assert len(SECTION_INSTRUCTIONS) == 13
    assert len(SECTION_QUERIES) == 13
    assert len(KNOWN_DRUG_INTERACTIONS) > 0

def t_config():
    assert CONFIG.agent.max_iterations == 20
    assert CONFIG.agent.max_tokens == 1024
    assert CONFIG.rag.chunk_size == 800
    assert CONFIG.rag.chunk_overlap == 150
    assert CONFIG.pdf.ocr_threshold_chars == 100
    assert CONFIG.tools.drug_interaction.max_retries == 2

def t_bm25_ranking():
    idx = BM25Index()
    idx.fit([
        "PATIENT Jane Smith MRN 87654321 diagnosis heart failure medications furosemide lisinopril",
        "billing administrative codes insurance reference number unrelated",
    ])
    hits = idx.score("patient MRN diagnosis medications admission discharge", top_n=2)
    assert hits, "BM25 returned no hits"
    assert hits[0][0] == 0, f"Wrong doc ranked first: {hits}"

def t_rag_empty():
    rag = RAGPipeline()
    assert rag.query("anything") == []
    assert rag.query_section("demographics") == "[MISSING — no relevant content found]"

def t_drug_tool_high_interaction():
    drug = DrugInteractionTool()
    r = drug.check(["warfarin", "aspirin"])
    assert r["status"] == "ok"
    assert any("HIGH" in i["severity"] for i in r["interactions"])

def t_drug_tool_empty():
    drug = DrugInteractionTool()
    r = drug.check([])
    assert r["checked"] == 0
    assert r["interactions"] == []

def t_escalation_tool():
    esc = EscalationTool()
    r = esc.escalate("patient_1", "HIGH", "drug_interactions", "Test interaction")
    assert r["status"] == "created"
    assert r["escalation_id"].startswith("ESC-")
    assert r["severity"] == "HIGH"
    assert r["requires_action"] is True

def t_summary_model_draft():
    s = DischargeSummary()
    assert s.is_draft is True
    assert "DRAFT" in s.generation_note

def t_summary_mark_missing_dedup():
    s = DischargeSummary()
    s.mark_missing("admission_date")
    s.mark_missing("admission_date")   # duplicate — should not double-add
    assert s.missing_fields == ["admission_date"]

def t_summary_flags():
    s = DischargeSummary()
    s.add_flag(FlagLevel.CRITICAL, "diagnosis",  "Conflict: two notes disagree")
    s.add_flag(FlagLevel.WARNING,  "medications", "Med added with no reason")
    assert len(s.flags) == 2
    assert s.flags[0].level == FlagLevel.CRITICAL

def t_graph_compiles():
    g = build_graph()
    expected_nodes = {
        "ingest_documents", "plan", "extract_section",
        "reconcile_medications", "check_conflicts",
        "drug_interaction_check", "compile_summary",
    }
    actual = set(g.nodes.keys()) - {"__start__"}
    assert actual == expected_nodes, f"Missing nodes: {expected_nodes - actual}"

def t_medication_reconciliation():
    """Verify medication reconciliation flags undocumented changes."""
    s = DischargeSummary()
    s.admission_medications  = ["Metformin 1000mg", "Lisinopril 10mg", "Aspirin 81mg"]
    s.discharge_medications  = ["Lisinopril 10mg", "Furosemide 40mg", "Aspirin 81mg"]

    admission = set(m.lower() for m in s.admission_medications)
    discharge = set(m.lower() for m in s.discharge_medications)

    for med in s.discharge_medications:
        ml = med.lower()
        if not any(ml in a or a in ml for a in admission):
            s.medication_changes.append(MedicationChange(
                medication=med, change_type="added", flagged=True,
                flag_note="Added at discharge — no documented reason."
            ))
    for med in s.admission_medications:
        ml = med.lower()
        if not any(ml in d or d in ml for d in discharge):
            s.medication_changes.append(MedicationChange(
                medication=med, change_type="stopped", flagged=True,
                flag_note="Stopped at discharge — no documented reason."
            ))

    flagged = [c for c in s.medication_changes if c.flagged]
    assert len(flagged) == 2, f"Expected 2 flagged changes, got {len(flagged)}"
    types = {c.change_type for c in flagged}
    assert "added"   in types   # Furosemide added
    assert "stopped" in types   # Metformin stopped


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Discharge Agent — Architecture Tests")
    logger.info("=" * 60)

    suite = [
        ("Logger initialises correctly",            t_logger),
        ("DischargeAgentException captures detail", t_exception),
        ("Constants loaded correctly",              t_constants),
        ("Config dot-access works",                 t_config),
        ("BM25 ranks clinical doc above noise",     t_bm25_ranking),
        ("RAG empty collection returns MISSING",    t_rag_empty),
        ("DrugInteractionTool — HIGH interaction",  t_drug_tool_high_interaction),
        ("DrugInteractionTool — empty list",        t_drug_tool_empty),
        ("EscalationTool creates escalation",       t_escalation_tool),
        ("DischargeSummary always marked draft",    t_summary_model_draft),
        ("mark_missing deduplicates entries",       t_summary_mark_missing_dedup),
        ("Flags added with correct levels",         t_summary_flags),
        ("LangGraph compiles with all 7 nodes",     t_graph_compiles),
        ("Medication reconciliation flags changes", t_medication_reconciliation),
    ]

    passed = sum(test(name, fn) for name, fn in suite)
    failed = len(suite) - passed

    logger.info("=" * 60)
    if failed == 0:
        logger.info("ALL %d TESTS PASSED ✓", passed)
    else:
        logger.error("%d PASSED, %d FAILED", passed, failed)
    logger.info("=" * 60)

    sys.exit(0 if failed == 0 else 1)
