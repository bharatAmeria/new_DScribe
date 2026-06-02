"""
Reward calculator for Part 2 — Learning from Doctor Edits.

Reward signal: how close is the agent's draft to the doctor's corrected version?
Higher reward = less editing needed = better draft.

Two metrics:
  1. Normalized edit distance (0=identical, 1=completely different)
     reward = 1 - edit_distance
  2. Section match rate (fraction of sections accepted unchanged)

Uses stdlib difflib — no extra dependencies.
"""
import sys
import logging
import difflib
from dataclasses import dataclass, field

from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


@dataclass
class SectionScore:
    section: str
    agent_text: str
    doctor_text: str
    edit_distance: float   # 0.0 = identical, 1.0 = completely different
    reward: float          # 1 - edit_distance
    unchanged: bool        # True if doctor made no edits


@dataclass
class IterationResult:
    iteration: int
    patient_id: str
    section_scores: list[SectionScore] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        if not self.section_scores:
            return 0.0
        return sum(s.reward for s in self.section_scores) / len(self.section_scores)

    @property
    def avg_edit_distance(self) -> float:
        if not self.section_scores:
            return 1.0
        return sum(s.edit_distance for s in self.section_scores) / len(self.section_scores)

    @property
    def section_match_rate(self) -> float:
        """Fraction of sections the doctor left unchanged."""
        if not self.section_scores:
            return 0.0
        return sum(1 for s in self.section_scores if s.unchanged) / len(self.section_scores)

    def summary(self) -> str:
        return (
            f"Iteration {self.iteration} | "
            f"Avg reward: {self.avg_reward:.3f} | "
            f"Avg edit distance: {self.avg_edit_distance:.3f} | "
            f"Section match rate: {self.section_match_rate:.1%}"
        )


class RewardCalculator:
    """
    Computes reward signal from (agent_draft, doctor_edited) pairs.
    Lower edit burden = higher reward.
    """

    def __init__(self):
        logger.info("RewardCalculator initialised")

    def score_sections(
        self,
        iteration: int,
        patient_id: str,
        agent_sections: dict[str, str],
        doctor_sections: dict[str, str],
        sections_to_score: list[str] | None = None,
    ) -> IterationResult:
        """
        Compare agent's extracted sections vs doctor's corrected versions.
        Returns an IterationResult with per-section and aggregate scores.
        """
        try:
            result = IterationResult(iteration=iteration, patient_id=patient_id)
            sections = sections_to_score or list(agent_sections.keys())

            for section in sections:
                agent_text  = str(agent_sections.get(section, "")).strip()
                doctor_text = str(doctor_sections.get(section, "")).strip()

                if not agent_text and not doctor_text:
                    continue

                edit_dist = self._normalized_edit_distance(agent_text, doctor_text)
                reward    = round(1.0 - edit_dist, 4)
                unchanged = edit_dist < 0.05   # <5% change = effectively unchanged

                result.section_scores.append(SectionScore(
                    section=section,
                    agent_text=agent_text,
                    doctor_text=doctor_text,
                    edit_distance=round(edit_dist, 4),
                    reward=reward,
                    unchanged=unchanged,
                ))

            logger.info(result.summary())
            return result

        except Exception as e:
            raise DischargeAgentException(e, sys)

    @staticmethod
    def _normalized_edit_distance(a: str, b: str) -> float:
        """
        Normalized edit distance using difflib SequenceMatcher.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        # SequenceMatcher ratio = 2*matches / (len(a) + len(b))
        ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return round(1.0 - ratio, 4)
