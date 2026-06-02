"""Mock clinical tools: drug interaction lookup and clinician escalation."""
import sys
import logging
import random
import time

from src.config import CONFIG
from src.constants import KNOWN_DRUG_INTERACTIONS
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


class DrugInteractionTool:
    """
    Mock drug-drug interaction checker.
    Simulates occasional tool failures to test agent robustness.
    """

    def __init__(self):
        cfg = CONFIG.tools.drug_interaction
        self.max_retries = cfg.max_retries
        self.failure_rate = cfg.failure_rate
        logging.info("DrugInteractionTool initialised (retries=%d, failure_rate=%.0f%%)",
                     self.max_retries, self.failure_rate * 100)

    def check(self, medications: list[str]) -> dict:
        """
        Check a list of medications for known interactions.
        Returns: {checked, interactions, errors, status}
        """
        try:
            interactions = []
            errors = []

            for attempt in range(self.max_retries + 1):
                # Simulate transient failure
                if attempt < self.max_retries and random.random() < self.failure_rate:
                    logging.warning(
                        "DrugInteractionTool: timeout on attempt %d — retrying...", attempt + 1
                    )
                    time.sleep(0.05)
                    errors.append(f"Timeout on attempt {attempt + 1}")
                    continue

                normalized = [m.lower().strip() for m in medications]
                for i, drug1 in enumerate(normalized):
                    for drug2 in normalized[i + 1:]:
                        pair = frozenset({drug1, drug2})
                        # Exact pair match
                        if pair in KNOWN_DRUG_INTERACTIONS:
                            interactions.append({
                                "drug1": drug1,
                                "drug2": drug2,
                                "severity": KNOWN_DRUG_INTERACTIONS[pair],
                            })
                            continue
                        # Partial keyword match
                        for key_pair, severity in KNOWN_DRUG_INTERACTIONS.items():
                            keys = list(key_pair)
                            if any(k in drug1 for k in keys) and any(
                                k in drug2 for k in [k for k in keys if k not in drug1]
                            ):
                                interactions.append({
                                    "drug1": drug1,
                                    "drug2": drug2,
                                    "severity": severity,
                                    "match_type": "partial",
                                })
                break  # success

            result = {
                "checked": len(medications),
                "interactions": interactions,
                "errors": errors,
                "status": "partial" if errors else "ok",
            }
            logging.info(
                "DrugInteractionTool: checked %d meds → %d interaction(s) found",
                len(medications), len(interactions),
            )
            return result
        except Exception as e:
            raise DischargeAgentException(e, sys)


class EscalationTool:
    """
    Mock clinician escalation tool.
    In production: creates a ticket / sends a notification.
    """

    def __init__(self):
        logging.info("EscalationTool initialised")

    def escalate(
        self,
        patient_id: str,
        severity: str,
        section: str,
        message: str,
        requires_action: bool = True,
    ) -> dict:
        """Create an escalation record and log it."""
        try:
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
            logging.warning(
                "[ESCALATION %s] %s — %s: %s", escalation_id, severity, section, message
            )
            return result
        except Exception as e:
            raise DischargeAgentException(e, sys)
