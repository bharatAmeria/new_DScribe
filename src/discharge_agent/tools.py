"""Mock clinical tools: drug interaction lookup and clinician escalation."""
from __future__ import annotations
import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Known interaction pairs (simplified mock database)
KNOWN_INTERACTIONS = {
    frozenset({"warfarin", "aspirin"}): "HIGH: Increased bleeding risk",
    frozenset({"warfarin", "ibuprofen"}): "HIGH: Increased bleeding risk",
    frozenset({"metformin", "contrast"}): "MODERATE: Hold metformin 48h before/after contrast",
    frozenset({"ssri", "tramadol"}): "HIGH: Serotonin syndrome risk",
    frozenset({"clopidogrel", "omeprazole"}): "MODERATE: Reduced clopidogrel efficacy",
    frozenset({"ace inhibitor", "potassium"}): "MODERATE: Hyperkalemia risk",
    frozenset({"digoxin", "amiodarone"}): "HIGH: Increased digoxin toxicity",
    frozenset({"lisinopril", "potassium"}): "MODERATE: Hyperkalemia risk",
    frozenset({"methotrexate", "nsaid"}): "HIGH: Increased methotrexate toxicity",
}


def drug_interaction_lookup(medications: list[str], max_retries: int = 2) -> dict:
    """
    Mock drug interaction lookup tool.
    Returns interactions found or empty list.
    Simulates occasional failures with retry logic.
    """
    interactions = []
    errors = []

    for attempt in range(max_retries + 1):
        # Simulate occasional tool failure (10% chance)
        if attempt < max_retries and random.random() < 0.1:
            logger.warning(f"Drug interaction tool timed out (attempt {attempt+1}), retrying...")
            time.sleep(0.1)
            errors.append(f"Timeout on attempt {attempt+1}")
            continue

        # Normalize medication names for matching
        normalized = [m.lower().strip() for m in medications]

        for i, drug1 in enumerate(normalized):
            for drug2 in normalized[i+1:]:
                # Check direct matches
                pair = frozenset({drug1, drug2})
                if pair in KNOWN_INTERACTIONS:
                    interactions.append({
                        "drug1": drug1,
                        "drug2": drug2,
                        "severity": KNOWN_INTERACTIONS[pair],
                    })
                    continue
                # Check partial keyword matches
                for key_pair, severity in KNOWN_INTERACTIONS.items():
                    keys = list(key_pair)
                    if (any(k in drug1 for k in keys) and any(k in drug2 for k in [k for k in keys if k not in drug1])):
                        interactions.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": severity,
                            "note": "partial match",
                        })
        break  # success

    return {
        "checked": len(medications),
        "interactions": interactions,
        "errors": errors,
        "status": "partial" if errors else "ok",
    }


def escalate_to_clinician(
    patient_id: str,
    severity: str,
    section: str,
    message: str,
    requires_action: bool = True,
) -> dict:
    """
    Mock escalation tool — flags an issue for clinician review.
    In production this would create a ticket/notification.
    """
    escalation_id = f"ESC-{patient_id[:4].upper()}-{int(time.time()) % 10000:04d}"
    result = {
        "escalation_id": escalation_id,
        "patient_id": patient_id,
        "severity": severity,
        "section": section,
        "message": message,
        "requires_action": requires_action,
        "status": "created",
    }
    logger.info(f"[ESCALATION] {escalation_id}: {severity} — {section}: {message}")
    return result
