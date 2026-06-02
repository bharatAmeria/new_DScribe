#!/usr/bin/env python3
"""
Dry-run integration test — runs the full LangGraph agent with a mock LLM.
No API key required. Validates the complete node sequence and output structure.

Usage: PYTHONPATH=src python3 tests/test_dry_run.py
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discharge_agent.pdf_loader import DocumentContent, PageContent
from discharge_agent.models import DischargeSummary, FlagLevel
from discharge_agent.rag import RAGPipeline

SAMPLE_NOTE_P1 = """
PATIENT: Jane Smith  DOB: 1962-07-04  MRN: 87654321
ADMISSION DATE: 2024-02-01  DISCHARGE DATE: 2024-02-07

PRINCIPAL DIAGNOSIS: Community-acquired pneumonia (CAP)
SECONDARY DIAGNOSES: Type 2 diabetes mellitus, Hypertension

ALLERGIES: Penicillin (anaphylaxis), Sulfa (rash)

HOSPITAL COURSE:
Patient presented with 5-day history of productive cough, fever 39.1C, and
right lower lobe consolidation on CXR. Started on IV ceftriaxone + azithromycin.
Transitioned to oral antibiotics on day 3. Blood glucose was poorly controlled
on admission (HbA1c 9.2%). Insulin regimen adjusted. Discharged in stable condition.

PROCEDURES: Chest X-ray x2, CT chest with contrast, Sputum culture

ADMISSION MEDICATIONS:
- Metformin 1000mg twice daily
- Amlodipine 10mg daily
- Lisinopril 10mg daily
- Aspirin 81mg daily

DISCHARGE MEDICATIONS:
- Amoxicillin-clavulanate 875mg twice daily x 5 days
- Metformin 1000mg twice daily
- Amlodipine 10mg daily
- Lisinopril 10mg daily
- Insulin glargine 10 units nightly (NEW)

PENDING RESULTS: Sputum culture and sensitivity (no growth at 48h, final pending)
Legionella urinary antigen (sent 2024-02-02)

FOLLOW-UP:
- Primary care physician in 1 week
- Pulmonology referral for repeat CXR in 6 weeks
- Endocrinology referral for diabetes optimization

DISCHARGE CONDITION: Stable, improved. Afebrile x 24 hours.
"""

# Patient 2 — intentionally has conflicts and missing data
SAMPLE_NOTE_P2_NOTE1 = """
PATIENT: Robert Johnson  DOB: 1948-11-22  MRN: 11223344
ADMISSION DATE: 2024-03-05

PRINCIPAL DIAGNOSIS: NSTEMI (Non-ST elevation myocardial infarction)
NOTE FROM DR. CHEN: Discharge date 2024-03-10. Discharge diagnosis: NSTEMI.
Patient underwent PCI with drug-eluting stent to LAD.
"""

SAMPLE_NOTE_P2_NOTE2 = """
PATIENT: Robert Johnson  MRN: 11223344
NOTE FROM DR. PATEL (conflicting): Discharge date 2024-03-11.
Discharge diagnosis: Unstable angina (revised opinion).
MEDICATIONS: Clopidogrel 75mg daily, Aspirin 81mg daily, Atorvastatin 80mg nightly.
PENDING: Troponin trend (FINAL PENDING), Echo results pending.
"""


def make_mock_llm(responses: dict) -> MagicMock:
    mock = MagicMock()
    def side_effect(messages):
        content = str(messages[-1].content).lower() if messages else ""
        for keyword, response in responses.items():
            if keyword.lower() in content:
                m = MagicMock()
                m.content = response
                return m
        m = MagicMock()
        m.content = "MISSING"
        return m
    mock.invoke.side_effect = side_effect
    return mock


def run_patient(patient_id: str, note_text: str | list, llm_responses: dict, output_dir: Path) -> dict:
    """Run agent for one patient with a mock LLM."""
    from discharge_agent.graph import build_graph
    from discharge_agent.rag import RAGPipeline
    from discharge_agent.tracer import AgentTracer
    import discharge_agent.nodes as nodes_mod

    # Build doc(s)
    if isinstance(note_text, str):
        docs = {"note": DocumentContent(
            path=Path("note.pdf"),
            pages=[PageContent(page_num=1, text=note_text, source="text")]
        )}
    else:
        docs = {
            f"note_{i}": DocumentContent(
                path=Path(f"note_{i}.pdf"),
                pages=[PageContent(page_num=1, text=t, source="text")]
            ) for i, t in enumerate(note_text)
        }

    rag = RAGPipeline(patient_id)
    rag.index_documents(docs)

    tracer = AgentTracer(patient_id, output_dir)
    mock_llm = make_mock_llm(llm_responses)

    nodes_mod._rag = rag
    nodes_mod._tracer = tracer
    nodes_mod._llm = mock_llm

    compiled = build_graph()

    with patch("discharge_agent.nodes.load_patient_folder", return_value=docs):
        final = compiled.invoke({
            "patient_id": patient_id,
            "patient_folder": str(output_dir),
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

    return final


def test_patient_1():
    print("\n" + "=" * 60)
    print("PATIENT 1: Jane Smith (text PDF, complete data)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        responses = {
            "demographics": '{"name": "Jane Smith", "dob": "1962-07-04", "mrn": "87654321"}',
            "admission_date": "2024-02-01",
            "discharge_date": "2024-02-07",
            "principal_diagnosis": "Community-acquired pneumonia (CAP)",
            "secondary_diagnoses": "Type 2 diabetes mellitus, Hypertension",
            "hospital_course": "Patient presented with productive cough and fever. Treated with IV ceftriaxone + azithromycin, transitioned to oral antibiotics. Insulin regimen adjusted for poorly controlled diabetes.",
            "procedures": "Chest X-ray x2, CT chest with contrast, Sputum culture",
            "admission_medications": "Metformin 1000mg twice daily\nAmlodipine 10mg daily\nLisinopril 10mg daily\nAspirin 81mg daily",
            "discharge_medications": "Amoxicillin-clavulanate 875mg twice daily x 5 days\nMetformin 1000mg twice daily\nAmlodipine 10mg daily\nLisinopril 10mg daily\nInsulin glargine 10 units nightly",
            "allergies": "Penicillin (anaphylaxis), Sulfa (rash)",
            "follow_up": "Primary care in 1 week\nPulmonology referral for repeat CXR in 6 weeks\nEndocrinology referral",
            "pending_results": "Sputum culture and sensitivity (final pending)\nLegionella urinary antigen (sent 2024-02-02)",
            "discharge_condition": "Stable, improved. Afebrile x 24 hours.",
            "conflict": "NO CONFLICT",
            "disagree": "NO CONFLICT",
        }

        final = run_patient("patient_1", SAMPLE_NOTE_P1, responses, out)
        s: DischargeSummary = final["summary"]

        assert s is not None
        assert s.is_draft is True
        assert s.patient_name == "Jane Smith"
        assert s.principal_diagnosis == "Community-acquired pneumonia (CAP)"
        assert len(s.discharge_medications) >= 1
        assert len(s.pending_results) >= 1

        # Insulin was added at discharge with no prior admission record → should be flagged
        flagged = [c for c in s.medication_changes if c.flagged]
        assert len(flagged) >= 1, "New medications should be flagged for reconciliation"

        print(f"  ✓ Patient: {s.patient_name}")
        print(f"  ✓ Principal Dx: {s.principal_diagnosis}")
        print(f"  ✓ Discharge meds: {len(s.discharge_medications)}")
        print(f"  ✓ Med changes flagged: {len(flagged)}")
        print(f"  ✓ Pending results: {len(s.pending_results)}")
        print(f"  ✓ Flags: {len(s.flags)}")
        print(f"  ✓ Missing fields: {s.missing_fields}")
        print(f"  ✓ Always draft: {s.is_draft}")

        # Print flagged medication changes
        for c in flagged:
            print(f"    → [FLAG] {c.change_type.upper()}: {c.medication}")
            print(f"      Reason: {c.flag_note}")

        return s


def test_patient_2_conflicts_and_missing():
    print("\n" + "=" * 60)
    print("PATIENT 2: Robert Johnson (conflicting notes, missing data)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        responses = {
            "demographics": '{"name": "Robert Johnson", "dob": "1948-11-22", "mrn": "11223344"}',
            "admission_date": "2024-03-05",
            "discharge_date": "CONFLICT: Note 1 says 2024-03-10 (Dr. Chen), Note 2 says 2024-03-11 (Dr. Patel)",
            # Two notes give different discharge diagnoses
            "principal_diagnosis": "CONFLICT: Dr. Chen documents NSTEMI; Dr. Patel documents Unstable Angina",
            "secondary_diagnoses": "MISSING",
            "hospital_course": "Patient underwent PCI with drug-eluting stent to LAD for NSTEMI. Clinical course documented inconsistently across notes.",
            "procedures": "PCI with drug-eluting stent to LAD",
            "admission_medications": "MISSING",
            "discharge_medications": "Clopidogrel 75mg daily\nAspirin 81mg daily\nAtorvastatin 80mg nightly",
            "allergies": "MISSING",
            "follow_up": "MISSING",
            "pending_results": "Troponin trend (FINAL PENDING)\nEchocardiogram results (pending)",
            "discharge_condition": "MISSING",
            "conflict": "CONFLICT: Discharge date differs between Dr. Chen (2024-03-10) and Dr. Patel (2024-03-11)",
            "disagree": "CONFLICT: discharge diagnosis differs — NSTEMI vs Unstable Angina",
        }

        final = run_patient(
            "patient_2",
            [SAMPLE_NOTE_P2_NOTE1, SAMPLE_NOTE_P2_NOTE2],
            responses,
            out,
        )
        s: DischargeSummary = final["summary"]

        assert s is not None
        assert s.is_draft is True

        # Conflicts should be flagged, NOT resolved
        critical_flags = [f for f in s.flags if f.level == FlagLevel.CRITICAL]
        assert len(critical_flags) >= 1, "Conflicts should create CRITICAL flags"

        # Missing fields should be recorded
        assert len(s.missing_fields) >= 1, "Missing data should be recorded"

        # Escalations should be created for critical issues
        assert len(final["escalation_log"]) >= 1, "Critical flags should trigger escalations"

        # Admission meds unknown → all discharge meds flagged as added
        flagged_meds = [c for c in s.medication_changes if c.flagged]
        assert len(flagged_meds) >= 1

        print(f"  ✓ Patient: {s.patient_name}")
        print(f"  ✓ Missing fields: {s.missing_fields}")
        print(f"  ✓ CRITICAL flags: {len(critical_flags)}")
        for f in critical_flags:
            print(f"    → [CRITICAL] {f.section}: {f.message[:80]}")
        print(f"  ✓ Escalations created: {len(final['escalation_log'])}")
        for e in final["escalation_log"]:
            print(f"    → {e['escalation_id']}: {e['severity']} — {e['section']}")
        print(f"  ✓ Flagged meds (no admission list): {len(flagged_meds)}")
        print(f"  ✓ Pending results: {len(s.pending_results)}")
        print(f"  ✓ Always draft: {s.is_draft}")

        return s


def test_step_cap():
    """Verify agent respects max_iterations and doesn't run forever."""
    print("\n" + "=" * 60)
    print("STEP CAP TEST: Agent must stop at max_iterations")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        note = "PATIENT: Test Patient. Minimal data."
        docs = {"note": DocumentContent(
            path=Path("note.pdf"),
            pages=[PageContent(page_num=1, text=note, source="text")]
        )}

        import discharge_agent.nodes as nodes_mod
        rag = RAGPipeline("cap_test")
        rag.index_documents(docs)
        tracer = AgentTracer("cap_test", out)
        mock_llm = make_mock_llm({"": "MISSING"})
        nodes_mod._rag = rag
        nodes_mod._tracer = tracer
        nodes_mod._llm = mock_llm

        from discharge_agent.graph import build_graph
        compiled = build_graph()

        with patch("discharge_agent.nodes.load_patient_folder", return_value=docs):
            final = compiled.invoke({
                "patient_id": "cap_test",
                "patient_folder": str(out),
                "iteration": 0,
                "max_iterations": 3,  # Very low cap
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

        assert final["iteration"] <= 4, f"Agent exceeded cap: {final['iteration']} iterations"
        print(f"  ✓ Agent stopped at iteration {final['iteration']} (cap=3)")
        print(f"  ✓ Stop reason: {final.get('stop_reason', 'cap hit')}")


if __name__ == "__main__":
    from discharge_agent.tracer import AgentTracer  # import here after path setup

    print("\n🏥 Discharge Summary Agent — Dry Run Tests")
    print("(Using mock LLM — no API key required)")

    try:
        s1 = test_patient_1()
        s2 = test_patient_2_conflicts_and_missing()
        test_step_cap()

        print("\n" + "=" * 60)
        print("✅ ALL DRY-RUN TESTS PASSED")
        print("=" * 60)
        print("\nTo run with a real LLM:")
        print("  1. Copy .env.example to .env and add ANTHROPIC_API_KEY")
        print("  2. Place patient PDFs in patients/patient_1/ etc.")
        print("  3. Run: PYTHONPATH=src python3 -m discharge_agent.main run patients/patient_1")
        print("     Or:  PYTHONPATH=src python3 -m discharge_agent.main batch patients/")

    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)
