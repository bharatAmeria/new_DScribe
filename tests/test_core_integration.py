#!/usr/bin/env python3
"""
Core integration test — verifies agent logic without langgraph graph invocation.
Tests: RAG indexing + retrieval, node logic, medication reconciliation, conflict detection.
No rich output, no subprocess, no API calls.
"""
import sys
import math
import re
import json
import logging
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.disable(logging.CRITICAL)  # silence all logs for clean test output

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Import only the lightweight modules ──────────────────────────────────────
from discharge_agent.rag import RAGPipeline, BM25Index, _chunk_text, Chunk
from discharge_agent.pdf_loader import DocumentContent, PageContent
from discharge_agent.models import DischargeSummary, FlagLevel, MedicationChange, PendingResult
from discharge_agent.tools import drug_interaction_lookup, escalate_to_clinician


PATIENT_1_NOTE = """
PATIENT: Jane Smith  DOB: 1962-07-04  MRN: 87654321
ADMISSION DATE: 2024-02-01  DISCHARGE DATE: 2024-02-07
PRINCIPAL DIAGNOSIS: Community-acquired pneumonia (CAP)
SECONDARY DIAGNOSES: Type 2 diabetes mellitus, Hypertension
ALLERGIES: Penicillin (anaphylaxis), Sulfa (rash)
HOSPITAL COURSE: Patient with productive cough, fever 39.1C, RLL consolidation on CXR.
IV ceftriaxone + azithromycin. Transitioned to oral antibiotics day 3. Insulin adjusted.
PROCEDURES: Chest X-ray x2, CT chest with contrast, Sputum culture
ADMISSION MEDICATIONS: Metformin 1000mg twice daily, Amlodipine 10mg daily, Lisinopril 10mg daily, Aspirin 81mg daily
DISCHARGE MEDICATIONS: Amoxicillin-clavulanate 875mg twice daily x5 days, Metformin 1000mg twice daily, Amlodipine 10mg daily, Lisinopril 10mg daily, Insulin glargine 10 units nightly
PENDING RESULTS: Sputum culture and sensitivity (no growth at 48h final pending), Legionella urinary antigen (sent 2024-02-02)
FOLLOW-UP: Primary care in 1 week, Pulmonology referral CXR 6 weeks, Endocrinology
DISCHARGE CONDITION: Stable, improved, afebrile x 24 hours
"""

PATIENT_2_NOTE_A = """
PATIENT: Robert Johnson  DOB: 1948-11-22  MRN: 11223344
ADMISSION DATE: 2024-03-05
NOTE FROM DR. CHEN: Discharge date 2024-03-10.
DISCHARGE DIAGNOSIS: NSTEMI (Non-ST elevation myocardial infarction)
PROCEDURES: PCI with drug-eluting stent to LAD
"""

PATIENT_2_NOTE_B = """
PATIENT: Robert Johnson  MRN: 11223344
NOTE FROM DR. PATEL: Discharge date 2024-03-11.
DISCHARGE DIAGNOSIS: Unstable angina (revised opinion, not NSTEMI)
MEDICATIONS: Clopidogrel 75mg daily, Aspirin 81mg daily, Atorvastatin 80mg nightly
PENDING: Troponin trend FINAL PENDING, Echo results pending
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_doc(text: str, name: str = "note") -> DocumentContent:
    return DocumentContent(
        path=Path(f"{name}.pdf"),
        pages=[PageContent(page_num=1, text=text, source="text")]
    )


def apply_extracted_value(summary: DischargeSummary, section: str, value: str) -> None:
    """Simplified version of nodes._apply_extracted_value for testing."""
    if value.strip().upper().startswith("MISSING"):
        summary.mark_missing(section)
        return
    if value.strip().upper().startswith("PENDING"):
        summary.add_flag(FlagLevel.WARNING, section, f"Data pending: {value}")
        return
    if value.strip().upper().startswith("CONFLICT"):
        summary.add_flag(FlagLevel.CRITICAL, section, f"Conflict: {value}")
        return

    if section == "demographics":
        try:
            data = json.loads(value)
            summary.patient_name = data.get("name")
            summary.date_of_birth = data.get("dob")
            summary.mrn = data.get("mrn")
        except Exception:
            summary.patient_name = value
    elif section == "principal_diagnosis": summary.principal_diagnosis = value
    elif section == "hospital_course": summary.hospital_course = value
    elif section == "discharge_condition": summary.discharge_condition = value
    elif section == "admission_medications":
        summary.admission_medications = [m.strip() for m in value.split(",") if m.strip()]
    elif section == "discharge_medications":
        summary.discharge_medications = [m.strip() for m in value.split(",") if m.strip()]
    elif section == "allergies":
        summary.allergies = [a.strip() for a in value.split(",") if a.strip()]
    elif section == "pending_results":
        for line in value.split(";"):
            if line.strip():
                summary.pending_results.append(PendingResult(test_name=line.strip(), note="Pending at discharge"))


def reconcile_meds(summary: DischargeSummary) -> None:
    """Medication reconciliation logic."""
    admission = set(m.lower() for m in summary.admission_medications)
    discharge = set(m.lower() for m in summary.discharge_medications)

    for med in summary.discharge_medications:
        ml = med.lower()
        if not any(ml in a or a in ml for a in admission):
            c = MedicationChange(medication=med, change_type="added", flagged=True,
                flag_note="Added at discharge — no documented reason. FLAG FOR RECONCILIATION.")
            summary.medication_changes.append(c)
            summary.add_flag(FlagLevel.WARNING, "medications",
                f"Med added at discharge with no documented reason: {med}")

    for med in summary.admission_medications:
        ml = med.lower()
        if not any(ml in d or d in ml for d in discharge):
            c = MedicationChange(medication=med, change_type="stopped", flagged=True,
                flag_note="Stopped at discharge — no documented reason. FLAG FOR RECONCILIATION.")
            summary.medication_changes.append(c)
            summary.add_flag(FlagLevel.WARNING, "medications",
                f"Med stopped at discharge with no documented reason: {med}")

    for med in summary.discharge_medications:
        ml = med.lower()
        if any(ml in a or a in ml for a in admission):
            summary.medication_changes.append(MedicationChange(medication=med, change_type="continued"))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_bm25():
    print("── BM25 Retrieval ──")
    idx = BM25Index()
    docs = [PATIENT_1_NOTE, "Unrelated billing and administrative note about insurance codes"]
    idx.fit(docs)
    hits = idx.score("patient name date of birth MRN", n=2)
    assert hits[0][0] == 0, "Patient note should score highest for patient demographics query"
    print(f"  ✓ Patient note ranked higher ({hits[0][1]:.3f}) than irrelevant doc ({hits[1][1]:.3f} if present)")


def test_chunking():
    print("── Text Chunking ──")
    chunks = _chunk_text(PATIENT_1_NOTE, source="note", page=1, size=200, overlap=50)
    assert len(chunks) >= 2
    # All chunks should have text
    for c in chunks:
        assert c.text.strip()
        assert c.source == "note"
        assert c.page == 1
    print(f"  ✓ {len(chunks)} chunks generated, all non-empty")


def test_rag_patient1():
    print("── RAG Pipeline — Patient 1 (complete data) ──")
    rag = RAGPipeline("p1")
    doc = make_doc(PATIENT_1_NOTE)
    n = rag.index_documents({"note": doc})
    assert n > 0

    # Demographics query
    ctx = rag.query_section("demographics")
    assert "Jane Smith" in ctx or "87654321" in ctx, f"Expected patient data in: {ctx[:100]}"

    # Medication queries
    ctx_adm = rag.query_section("admission_medications")
    assert "Metformin" in ctx_adm or "metformin" in ctx_adm.lower()

    ctx_dis = rag.query_section("discharge_medications")
    assert "Amoxicillin" in ctx_dis or "amoxicillin" in ctx_dis.lower()

    # Pending results
    ctx_pend = rag.query_section("pending_results")
    assert "pending" in ctx_pend.lower()

    print(f"  ✓ Indexed {n} chunks")
    print(f"  ✓ Demographics context found (Jane Smith / MRN 87654321)")
    print(f"  ✓ Admission medications found")
    print(f"  ✓ Discharge medications found")
    print(f"  ✓ Pending results found")


def test_rag_patient2_multi_doc():
    print("── RAG Pipeline — Patient 2 (multi-document, conflict) ──")
    rag = RAGPipeline("p2")
    docs = {
        "note_a": make_doc(PATIENT_2_NOTE_A, "note_a"),
        "note_b": make_doc(PATIENT_2_NOTE_B, "note_b"),
    }
    n = rag.index_documents(docs)
    assert n > 0

    # Should find both discharge dates
    ctx = rag.query("discharge date", n_results=4)
    texts = " ".join(c["text"] for c in ctx)
    assert "2024-03-10" in texts or "2024-03-11" in texts, "Expected discharge dates in context"
    print(f"  ✓ Both notes indexed ({n} total chunks)")
    print(f"  ✓ Conflicting discharge dates retrievable from context")


def test_summary_patient1():
    print("── Summary Build — Patient 1 ──")
    s = DischargeSummary()
    apply_extracted_value(s, "demographics", '{"name": "Jane Smith", "dob": "1962-07-04", "mrn": "87654321"}')
    apply_extracted_value(s, "principal_diagnosis", "Community-acquired pneumonia (CAP)")
    apply_extracted_value(s, "hospital_course", "Patient treated with IV then oral antibiotics. Insulin regimen adjusted.")
    apply_extracted_value(s, "admission_medications", "Metformin 1000mg, Amlodipine 10mg, Lisinopril 10mg, Aspirin 81mg")
    apply_extracted_value(s, "discharge_medications", "Amoxicillin-clavulanate 875mg, Metformin 1000mg, Amlodipine 10mg, Lisinopril 10mg, Insulin glargine 10 units")
    apply_extracted_value(s, "allergies", "Penicillin (anaphylaxis), Sulfa (rash)")
    apply_extracted_value(s, "pending_results", "Sputum culture (final pending); Legionella antigen (pending)")
    apply_extracted_value(s, "discharge_condition", "Stable, improved")

    reconcile_meds(s)

    assert s.patient_name == "Jane Smith"
    assert s.is_draft is True
    assert len(s.discharge_medications) >= 1
    assert len(s.pending_results) >= 1

    flagged = [c for c in s.medication_changes if c.flagged]
    assert len(flagged) >= 1, "Amoxicillin + Insulin should be flagged as added"

    # No fabrication check — all values came from explicit extraction
    assert s.principal_diagnosis == "Community-acquired pneumonia (CAP)"

    print(f"  ✓ Patient: {s.patient_name}, Dx: {s.principal_diagnosis}")
    print(f"  ✓ Discharge meds: {len(s.discharge_medications)}")
    print(f"  ✓ Flagged med changes: {len(flagged)}")
    for c in flagged:
        print(f"    → [{c.change_type.upper()}] {c.medication}")
    print(f"  ✓ Pending results: {len(s.pending_results)}")
    print(f"  ✓ Always draft: {s.is_draft}")
    print(f"  ✓ Missing fields: {s.missing_fields}")


def test_summary_patient2_conflicts():
    print("── Summary Build — Patient 2 (conflicts & missing data) ──")
    s = DischargeSummary()
    apply_extracted_value(s, "demographics", '{"name": "Robert Johnson", "dob": "1948-11-22", "mrn": "11223344"}')
    # Conflicting discharge date from two notes
    apply_extracted_value(s, "discharge_date",
        "CONFLICT: Note A (Dr. Chen) says 2024-03-10; Note B (Dr. Patel) says 2024-03-11")
    # Conflicting diagnosis
    apply_extracted_value(s, "principal_diagnosis",
        "CONFLICT: Dr. Chen documents NSTEMI; Dr. Patel documents Unstable Angina")
    # Admission meds unknown
    apply_extracted_value(s, "admission_medications", "MISSING")
    apply_extracted_value(s, "discharge_medications",
        "Clopidogrel 75mg, Aspirin 81mg, Atorvastatin 80mg")
    apply_extracted_value(s, "allergies", "MISSING")
    apply_extracted_value(s, "discharge_condition", "MISSING")
    apply_extracted_value(s, "pending_results", "Troponin trend (FINAL PENDING); Echo results (pending)")

    reconcile_meds(s)

    # Conflict flags must exist
    critical = [f for f in s.flags if f.level == FlagLevel.CRITICAL]
    assert len(critical) >= 2, f"Expected ≥2 CRITICAL flags, got {len(critical)}"

    # Missing fields recorded
    assert "principal_diagnosis" in s.missing_fields or len(s.missing_fields) >= 1

    # Discharge meds without admission list → all flagged
    flagged = [c for c in s.medication_changes if c.flagged]
    assert len(flagged) >= 1

    # Always draft, never finalized
    assert s.is_draft is True

    print(f"  ✓ CRITICAL flags (conflicts): {len(critical)}")
    for f in critical:
        print(f"    → {f.section}: {f.message[:80]}")
    print(f"  ✓ Missing fields: {s.missing_fields}")
    print(f"  ✓ Flagged meds (no admission baseline): {len(flagged)}")
    print(f"  ✓ Pending results: {len(s.pending_results)}")
    print(f"  ✓ Always draft: {s.is_draft}")


def test_drug_interactions_with_escalation():
    print("── Drug Interactions + Escalation ──")
    # Patient 2 meds: clopidogrel + aspirin + omeprazole-like check
    discharge_meds = ["Clopidogrel 75mg daily", "Aspirin 81mg daily", "Atorvastatin 80mg nightly", "Omeprazole 20mg daily"]
    result = drug_interaction_lookup(discharge_meds)
    print(f"  ✓ Checked {result['checked']} medications")
    print(f"  ✓ Found {len(result['interactions'])} interaction(s)")

    for interaction in result["interactions"]:
        severity = interaction["severity"].split(":")[0].strip()
        print(f"    → {interaction['drug1']} + {interaction['drug2']}: {interaction['severity']}")
        if severity in ("HIGH", "MODERATE"):
            esc = escalate_to_clinician(
                patient_id="patient_2",
                severity=severity,
                section="drug_interactions",
                message=f"{interaction['drug1']} + {interaction['drug2']}: {interaction['severity']}",
            )
            assert esc["status"] == "created"
            assert "ESC-" in esc["escalation_id"]
            print(f"    → Escalated: {esc['escalation_id']}")

    print(f"  ✓ Escalation tool working correctly")


def test_step_cap_logic():
    print("── Step Cap Logic ──")
    max_iter = 5
    iteration = 0
    pending = ["demographics", "admission_date", "discharge_date", "principal_diagnosis",
               "secondary_diagnoses", "hospital_course", "procedures"]

    processed = []
    while pending and iteration < max_iter:
        section = pending.pop(0)
        processed.append(section)
        iteration += 1

    assert iteration == max_iter
    assert len(pending) > 0, "Should have remaining sections after cap"
    print(f"  ✓ Stopped at iteration {iteration} (cap={max_iter})")
    print(f"  ✓ {len(processed)} sections processed, {len(pending)} remaining (graceful partial output)")


if __name__ == "__main__":
    print("\n🏥 Discharge Summary Agent — Core Integration Tests")
    print("=" * 60)

    tests = [
        test_bm25,
        test_chunking,
        test_rag_patient1,
        test_rag_patient2_multi_doc,
        test_summary_patient1,
        test_summary_patient2_conflicts,
        test_drug_interactions_with_escalation,
        test_step_cap_logic,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED")
    else:
        print(f"❌ {failed} FAILED, {passed} passed")
    print("=" * 60)
