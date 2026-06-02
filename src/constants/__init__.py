# ── Pipeline Stage Names ────────────────────────────────────────────────────
PDF_INGESTION_STAGE_NAME    = "PDF Ingestion"
RAG_INDEXING_STAGE_NAME     = "RAG Indexing"
AGENT_RUN_STAGE_NAME        = "Agent Run"
LEARNING_LOOP_STAGE_NAME    = "Learning Loop"

# ── Required Python version ─────────────────────────────────────────────────
REQUIRED_PYTHON = "python3"

# ── Discharge Summary Sections ───────────────────────────────────────────────
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

# ── Section extraction instructions (used by LLM nodes) ─────────────────────
SECTION_INSTRUCTIONS: dict[str, str] = {
    "demographics": (
        "Extract patient full name, date of birth, and MRN/patient ID "
        "as a JSON object with keys: name, dob, mrn."
    ),
    "admission_date": (
        "Extract the hospital admission date (YYYY-MM-DD if possible, otherwise as written)."
    ),
    "discharge_date": (
        "Extract the hospital discharge date (YYYY-MM-DD if possible, otherwise as written)."
    ),
    "principal_diagnosis": "Extract the single principal/primary diagnosis.",
    "secondary_diagnoses": "Extract all secondary diagnoses as a comma-separated list.",
    "hospital_course": (
        "Summarize the hospital course in 2-4 sentences covering key events, "
        "treatments, and patient response."
    ),
    "procedures": "List all procedures and surgical interventions performed during admission.",
    "admission_medications": "List all medications the patient was taking on admission (home meds).",
    "discharge_medications": "List all medications prescribed at discharge, with doses and frequencies.",
    "allergies": "List all documented allergies and reactions. If NKDA, state that.",
    "follow_up": "Extract all follow-up instructions, appointments, and referrals.",
    "pending_results": "List any lab results, cultures, or studies still pending/outstanding.",
    "discharge_condition": "State the patient's condition at discharge (e.g., stable, improved, critical).",
}

# ── Section RAG query templates ──────────────────────────────────────────────
SECTION_QUERIES: dict[str, str] = {
    "demographics": "patient name date of birth MRN medical record number age gender",
    "admission_date": "admission date admitted hospital",
    "discharge_date": "discharge date discharged",
    "principal_diagnosis": "principal diagnosis primary diagnosis main diagnosis",
    "secondary_diagnoses": "secondary diagnosis comorbidities other diagnoses",
    "hospital_course": "hospital course treatment clinical course summary events",
    "procedures": "procedures operations surgery interventions performed",
    "admission_medications": "admission medications home medications on admission",
    "discharge_medications": "discharge medications on discharge prescribed",
    "allergies": "allergies allergic reactions drug allergy NKDA no known",
    "follow_up": "follow up instructions outpatient clinic appointment referral",
    "pending_results": "pending results awaiting outstanding labs cultures",
    "discharge_condition": "discharge condition stable improved critical",
}

# ── Known drug interactions (mock DB) ────────────────────────────────────────
KNOWN_DRUG_INTERACTIONS: dict[frozenset, str] = {
    frozenset({"warfarin", "aspirin"}):          "HIGH: Increased bleeding risk",
    frozenset({"warfarin", "ibuprofen"}):         "HIGH: Increased bleeding risk",
    frozenset({"ssri", "tramadol"}):              "HIGH: Serotonin syndrome risk",
    frozenset({"digoxin", "amiodarone"}):         "HIGH: Increased digoxin toxicity",
    frozenset({"methotrexate", "nsaid"}):         "HIGH: Increased methotrexate toxicity",
    frozenset({"metformin", "contrast"}):         "MODERATE: Hold metformin 48h around contrast",
    frozenset({"clopidogrel", "omeprazole"}):     "MODERATE: Reduced clopidogrel efficacy",
    frozenset({"ace inhibitor", "potassium"}):    "MODERATE: Hyperkalemia risk",
    frozenset({"lisinopril", "potassium"}):       "MODERATE: Hyperkalemia risk",
}
