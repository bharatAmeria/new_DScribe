"""
Correction memory for Part 2 — Learning from Doctor Edits.

Stores (agent_wrote, doctor_corrected) pairs per section across iterations.
On future runs, injects relevant past corrections into the LLM prompt as
few-shot examples — enabling the agent to learn from doctor feedback
without any fine-tuning.

Persisted to disk as JSON so memory accumulates across runs.
"""
import sys
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime

from src.config import CONFIG
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    section:     str
    iteration:   int
    patient_id:  str
    agent_wrote: str
    doctor_wrote: str
    edit_distance: float
    reward:      float
    timestamp:   str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def was_improved(self) -> bool:
        return self.edit_distance > 0.05   # doctor made meaningful edits


class CorrectionMemory:
    """
    Persistent correction memory.
    - Stores all (agent, doctor) pairs per section.
    - Provides formatted few-shot examples for LLM prompts.
    - Saves to artifacts/memory/{patient_id}_memory.json.
    """

    def __init__(self, patient_id: str):
        self._patient_id  = patient_id
        self._memory_dir  = Path(CONFIG.learning.memory_dir)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._path        = self._memory_dir / f"{patient_id}_memory.json"
        self._corrections: list[Correction] = []
        self._load()
        logger.info(
            "CorrectionMemory loaded: %d past corrections for patient '%s'",
            len(self._corrections), patient_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, corrections: list[Correction]) -> None:
        """Persist a batch of new corrections to memory."""
        try:
            self._corrections.extend(corrections)
            self._save()
            logger.info(
                "Stored %d correction(s) → memory now has %d total",
                len(corrections), len(self._corrections),
            )
        except Exception as e:
            raise DischargeAgentException(e, sys)

    def get_prompt_injection(self, sections: list[str], max_per_section: int = 2) -> str:
        """
        Build a few-shot correction block to inject into the LLM prompt.
        Only includes corrections where the doctor made meaningful edits.
        """
        if not self._corrections:
            return ""

        lines = [
            "\n=== CORRECTION MEMORY (learn from past doctor edits) ===",
            "The doctor previously corrected these sections. Apply the same standards:\n",
        ]

        for section in sections:
            relevant = [
                c for c in self._corrections
                if c.section == section and c.was_improved
            ]
            # Most recent first
            relevant.sort(key=lambda c: c.timestamp, reverse=True)
            for c in relevant[:max_per_section]:
                lines.append(f"[{section.upper()}]")
                lines.append(f"  Agent wrote : {c.agent_wrote[:200]}")
                lines.append(f"  Doctor wrote: {c.doctor_wrote[:200]}")
                lines.append(f"  Edit distance: {c.edit_distance:.2f} | Reward: {c.reward:.2f}\n")

        if len(lines) == 2:   # no meaningful corrections found
            return ""

        lines.append("=== END CORRECTION MEMORY ===\n")
        return "\n".join(lines)

    def section_stats(self) -> dict[str, dict]:
        """Return per-section average reward and edit distance across all iterations."""
        stats: dict[str, dict] = {}
        for c in self._corrections:
            if c.section not in stats:
                stats[c.section] = {"rewards": [], "edit_distances": [], "count": 0}
            stats[c.section]["rewards"].append(c.reward)
            stats[c.section]["edit_distances"].append(c.edit_distance)
            stats[c.section]["count"] += 1

        return {
            section: {
                "avg_reward":        round(sum(v["rewards"]) / len(v["rewards"]), 3),
                "avg_edit_distance": round(sum(v["edit_distances"]) / len(v["edit_distances"]), 3),
                "count":             v["count"],
            }
            for section, v in stats.items()
        }

    def all_iterations(self) -> list[int]:
        return sorted(set(c.iteration for c in self._corrections))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump([asdict(c) for c in self._corrections], f, indent=2)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._corrections = [Correction(**d) for d in data]
        except Exception as e:
            logger.warning("Could not load correction memory from %s: %s", self._path, e)
            self._corrections = []
