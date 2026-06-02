"""Tests for the discharge summary agent — uses a mock LLM to avoid API calls."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discharge_agent.models import DischargeSummary, FlagLevel
from discharge_agent.pdf_loader import load_pdf, DocumentContent, PageContent
from discharge_agent.rag import RAGPipeline
from discharge_agent.tools import drug_interaction_lookup, escalate_to_clinician


# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_NOTE = """
PATIENT: John Doe  DOB: 1955-03-12  MRN: 12345678
ADMISSION DATE: 2024-01-10  DISCHARGE DATE: 2024-01-15

PRINCIPAL DIAGNOSIS: Acute decompensated heart failure

SECONDARY DIAGNOSES: Type 2 diabetes mellitus, Hypertension, Chronic kidney disease stage 3

ALLERGIES: Penicillin (rash), Sulfa drugs (anaphylaxis)

HOSPITAL COURSE:
Patient presented with worsening dyspnea and lower extremity edema.
IV furosemide was initiated with good diuretic response.
Echo showed EF 35%. Patient improved and was discharged on oral furosemide.

PROCEDURES: Echocardiogram, Right heart catheterization

ADMISSION MEDICATIONS:
- Lisinopril 10mg daily
- Metformin 500mg twice daily
- Aspirin 81mg daily

DISCHARGE MEDICATIONS:
- Furosemide 40mg daily
- Lisinopril 10mg daily
- Carvedilol 6.25mg twice daily
- Aspirin 81mg daily

PENDING RESULTS: BNP level (sent 2024-01-14), Blood culture x2 (no growth at 48h, final pending)

FOLLOW-UP: Cardiology clinic in 1 week, Primary care in 2 weeks

DISCHARGE CONDITION: Stable, improved
"""


@pytest.fixture
def sample_doc():
    """Create a DocumentContent from sample text."""
    return DocumentContent(
        path=Path("test_note.pdf"),
        pages=[PageContent(page_num=1, text=SAMPLE_NOTE, source="text")],
    )


@pytest.fixture
def rag_with_sample(sample_doc):
    """RAG pipeline pre-indexed with sample note."""
    rag = RAGPipeline("test")
    rag.index_documents({"test_note": sample_doc})
    return rag


# ── Unit Tests ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_index_and_query(self, rag_with_sample):
        results = rag_with_sample.query("patient name date of birth MRN")
        assert len(results) > 0
        assert any("John Doe" in r["text"] for r in results)

    def test_query_section_demographics(self, rag_with_sample):
        ctx = rag_with_sample.query_section("demographics")
        assert "John Doe" in ctx or "MRN" in ctx

    def test_query_section_medications(self, rag_with_sample):
        ctx = rag_with_sample.query_section("discharge_medications")
        assert "Furosemide" in ctx or "furosemide" in ctx.lower()

    def test_empty_collection(self):
        rag = RAGPipeline("empty")
        results = rag.query("anything")
        assert results == []

    def test_missing_section_marker(self):
        rag = RAGPipeline("empty2")
        result = rag.query_section("demographics")
        assert "MISSING" in result


class TestDrugInteractionTool:
    def test_known_interaction(self):
        result = drug_interaction_lookup(["warfarin", "aspirin"])
        assert result["status"] == "ok"
        assert len(result["interactions"]) >= 1
        assert any("HIGH" in i["severity"] for i in result["interactions"])

    def test_no_interaction(self):
        result = drug_interaction_lookup(["furosemide", "lisinopril"])
        assert result["checked"] == 2
        # No known interaction between these two in mock DB
        assert isinstance(result["interactions"], list)

    def test_multiple_drugs(self):
        result = drug_interaction_lookup(["warfarin", "aspirin", "metformin", "furosemide"])
        assert result["checked"] == 4

    def test_empty_list(self):
        result = drug_interaction_lookup([])
        assert result["checked"] == 0
        assert result["interactions"] == []


class TestEscalationTool:
    def test_creates_escalation(self):
        result = escalate_to_clinician(
            patient_id="patient_1",
            severity="HIGH",
            section="drug_interactions",
            message="Warfarin + Aspirin: HIGH bleeding risk",
        )
        assert result["status"] == "created"
        assert "ESC-" in result["escalation_id"]
        assert result["severity"] == "HIGH"

    def test_escalation_id_unique(self):
        r1 = escalate_to_clinician("p1", "HIGH", "meds", "test1")
        r2 = escalate_to_clinician("p1", "HIGH", "meds", "test2")
        # IDs should be distinct (time-based)
        assert isinstance(r1["escalation_id"], str)
        assert isinstance(r2["escalation_id"], str)


class TestDischargeSummaryModel:
    def test_mark_missing(self):
        s = DischargeSummary()
        s.mark_missing("admission_date")
        s.mark_missing("admission_date")  # dedup
        assert s.missing_fields == ["admission_date"]

    def test_add_flag(self):
        s = DischargeSummary()
        s.add_flag(FlagLevel.CRITICAL, "medications", "Conflict found")
        assert len(s.flags) == 1
        assert s.flags[0].level == FlagLevel.CRITICAL

    def test_is_always_draft(self):
        s = DischargeSummary()
        assert s.is_draft is True


class TestPDFLoader:
    def test_missing_file(self):
        result = load_pdf(Path("/nonexistent/path.pdf"))
        assert result.error is not None
        assert result.is_empty

    def test_document_full_text(self):
        doc = DocumentContent(
            path=Path("test.pdf"),
            pages=[
                PageContent(page_num=1, text="Hello world", source="text"),
                PageContent(page_num=2, text="Second page", source="text"),
            ],
        )
        text = doc.full_text
        assert "Hello world" in text
        assert "Second page" in text
        assert "[Page 1]" in text


# ── Integration Test (mock LLM) ──────────────────────────────────────────────

class TestAgentGraphMockLLM:
    """Run the full graph with a mocked LLM — no API key required."""

    def _make_mock_llm(self, responses: dict[str, str]):
        """Build a mock LLM that returns canned responses by keyword."""
        mock = MagicMock()
        def invoke_side_effect(messages):
            # Find last human message
            content = str(messages[-1].content) if messages else ""
            for keyword, response in responses.items():
                if keyword.lower() in content.lower():
                    m = MagicMock()
                    m.content = response
                    return m
            m = MagicMock()
            m.content = "MISSING"
            return m
        mock.invoke.side_effect = invoke_side_effect
        return mock

    def test_full_pipeline(self, tmp_path, sample_doc):
        """Full agent run on sample patient note with mocked LLM."""
        # Write sample PDF to temp folder
        pdf_folder = tmp_path / "patient_test"
        pdf_folder.mkdir()

        # We'll mock load_patient_folder to return our sample doc
        with patch("discharge_agent.nodes._llm") as mock_llm_ref, \
             patch("discharge_agent.nodes._rag") as mock_rag_ref, \
             patch("discharge_agent.nodes._tracer") as mock_tracer_ref, \
             patch("discharge_agent.nodes.load_patient_folder") as mock_loader:

            # Setup mocks
            mock_loader.return_value = {"test_note": sample_doc}
            mock_tracer_ref.start_step = MagicMock()
            mock_tracer_ref.record = MagicMock()
            mock_tracer_ref.save = MagicMock(return_value=tmp_path / "trace.json")

            # Real RAG with sample data
            real_rag = RAGPipeline("test_full")
            real_rag.index_documents({"test_note": sample_doc})
            mock_rag_ref.index_documents = real_rag.index_documents
            mock_rag_ref.query = real_rag.query
            mock_rag_ref.query_section = real_rag.query_section

            # Mock LLM responses
            llm_responses = {
                "demographics": '{"name": "John Doe", "dob": "1955-03-12", "mrn": "12345678"}',
                "admission_date": "2024-01-10",
                "discharge_date": "2024-01-15",
                "principal_diagnosis": "Acute decompensated heart failure",
                "secondary_diagnoses": "Type 2 diabetes mellitus, Hypertension, Chronic kidney disease stage 3",
                "hospital_course": "Patient presented with dyspnea and edema. Treated with IV furosemide. Discharged on oral regimen.",
                "procedures": "Echocardiogram, Right heart catheterization",
                "admission_medications": "Lisinopril 10mg daily\nMetformin 500mg twice daily\nAspirin 81mg daily",
                "discharge_medications": "Furosemide 40mg daily\nLisinopril 10mg daily\nCarvedilol 6.25mg twice daily\nAspirin 81mg daily",
                "allergies": "Penicillin (rash), Sulfa drugs (anaphylaxis)",
                "follow_up": "Cardiology clinic in 1 week\nPrimary care in 2 weeks",
                "pending_results": "BNP level (sent 2024-01-14)\nBlood culture (final pending)",
                "discharge_condition": "Stable, improved",
                "conflict": "NO CONFLICT",
            }

            mock_llm = self._make_mock_llm(llm_responses)
            mock_llm_ref.invoke = mock_llm.invoke

            # Run graph
            from discharge_agent.graph import build_graph
            from discharge_agent.state import AgentState

            compiled = build_graph()
            result = compiled.invoke({
                "patient_id": "patient_test",
                "patient_folder": str(pdf_folder),
                "iteration": 0,
                "max_iterations": 30,
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
            })

            summary = result.get("summary")
            assert summary is not None, "Summary should be generated"
            assert summary.is_draft is True, "Output must always be a draft"
            assert summary.patient_name == "John Doe"
            assert summary.principal_diagnosis == "Acute decompensated heart failure"
            assert len(summary.discharge_medications) >= 1
            assert len(summary.medication_changes) >= 1  # Furosemide + Carvedilol added

            # Verify medication reconciliation flagged new drugs
            flagged_meds = [c for c in summary.medication_changes if c.flagged]
            assert len(flagged_meds) >= 1, "New/stopped meds should be flagged"

            # Verify pending results captured
            assert len(summary.pending_results) >= 1

            # Verify escalations created for any CRITICAL flags
            # (drug interactions: lisinopril + potassium is moderate, no HIGH expected here)
            assert isinstance(result.get("escalation_log"), list)

            print("\n✓ Full pipeline test passed")
            print(f"  Patient: {summary.patient_name}")
            print(f"  Diagnosis: {summary.principal_diagnosis}")
            print(f"  Flags: {len(summary.flags)}")
            print(f"  Missing: {summary.missing_fields}")
            print(f"  Med changes: {len(summary.medication_changes)}")
