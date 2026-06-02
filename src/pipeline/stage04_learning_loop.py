"""
Stage 4 — Learning Loop (Part 2).

Orchestrates N iterations of:
  1. Agent generates discharge summary draft
  2. MockDoctor reviews and corrects the draft
  3. RewardCalculator scores the (draft, corrected) pair
  4. CorrectionMemory stores the edits
  5. Next iteration: agent sees past corrections → generates better draft

Tracks improvement curve and saves metrics report.
"""
import sys
import json
import logging
from pathlib import Path

from src.constants import LEARNING_LOOP_STAGE_NAME
from src.config import CONFIG
from src.exception import DischargeAgentException
from src.agent.graph import run_agent
from src.components.mock_doctor import MockDoctor
from src.components.reward_calculator import RewardCalculator, IterationResult
from src.components.correction_memory import CorrectionMemory, Correction
from src.components.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class LearningLoopPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main(
        patient_id: str,
        patient_folder: Path,
        pre_indexed_rag: RAGPipeline,
        n_iterations: int | None = None,
    ) -> dict:
        """
        Run the full learning loop for a patient.
        Returns: {iteration_results, final_reward, improvement, metrics_path}
        """
        try:
            n_iter     = n_iterations or CONFIG.learning.iterations
            output_dir = Path(CONFIG.patients.output_dir)
            metrics_dir= Path(CONFIG.learning.metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            reward_threshold    = CONFIG.learning.reward_threshold
            held_out_sections   = list(CONFIG.learning.held_out_sections)

            memory    = CorrectionMemory(patient_id)
            doctor    = MockDoctor()
            scorer    = RewardCalculator()
            results:  list[IterationResult] = []

            logger.info(
                "Learning loop: %d iterations for patient '%s' | held-out: %s",
                n_iter, patient_id, held_out_sections,
            )

            for iteration in range(1, n_iter + 1):
                logger.info("── Iteration %d / %d ──", iteration, n_iter)

                # ── Step 1: Agent generates draft ─────────────────────────────
                iter_output_dir = output_dir / f"iter_{iteration}"
                result = run_agent(
                    patient_id=patient_id,
                    patient_folder=patient_folder,
                    output_dir=iter_output_dir,
                    pre_indexed_rag=pre_indexed_rag,
                    correction_memory=memory,
                )

                summary = result.get("summary")
                if summary is None:
                    logger.error("Iteration %d: agent returned no summary — skipping", iteration)
                    continue

                # Extract sections as flat strings for comparison
                agent_sections = _summary_to_sections(summary)

                # ── Step 2: Mock doctor reviews ───────────────────────────────
                logger.info("MockDoctor reviewing iteration %d draft...", iteration)
                doctor_sections = doctor.review(agent_sections)

                # Save doctor's corrected version
                corrected_path = iter_output_dir / f"doctor_corrected_{patient_id}.json"
                corrected_path.parent.mkdir(parents=True, exist_ok=True)
                with open(corrected_path, "w") as f:
                    json.dump(doctor_sections, f, indent=2)

                # ── Step 3: Score the pair ────────────────────────────────────
                iter_result = scorer.score_sections(
                    iteration=iteration,
                    patient_id=patient_id,
                    agent_sections=agent_sections,
                    doctor_sections=doctor_sections,
                    sections_to_score=held_out_sections,
                )
                results.append(iter_result)

                logger.info(
                    "Iteration %d | reward=%.3f | edit_distance=%.3f | match_rate=%.1%%",
                    iteration,
                    iter_result.avg_reward,
                    iter_result.avg_edit_distance,
                    iter_result.section_match_rate * 100,
                )

                # ── Step 4: Store corrections in memory ───────────────────────
                new_corrections = [
                    Correction(
                        section=s.section,
                        iteration=iteration,
                        patient_id=patient_id,
                        agent_wrote=s.agent_text,
                        doctor_wrote=s.doctor_text,
                        edit_distance=s.edit_distance,
                        reward=s.reward,
                    )
                    for s in iter_result.section_scores
                    if s.was_improved
                ]
                if new_corrections:
                    memory.store(new_corrections)
                    logger.info(
                        "Stored %d correction(s) to memory (%d improved sections)",
                        len(new_corrections), len(new_corrections),
                    )
                else:
                    logger.info("No corrections needed — all sections accepted by doctor")

                # ── Early stopping ────────────────────────────────────────────
                if iter_result.avg_reward >= reward_threshold:
                    logger.info(
                        "Early stop at iteration %d: reward %.3f >= threshold %.3f",
                        iteration, iter_result.avg_reward, reward_threshold,
                    )
                    break

            # ── Save metrics report ───────────────────────────────────────────
            metrics_path = _save_metrics(patient_id, results, metrics_dir, held_out_sections)

            first_reward = results[0].avg_reward  if results else 0.0
            last_reward  = results[-1].avg_reward if results else 0.0
            improvement  = round(last_reward - first_reward, 4)

            logger.info(
                "Learning loop complete | iterations=%d | reward: %.3f → %.3f (Δ%.3f)",
                len(results), first_reward, last_reward, improvement,
            )

            return {
                "iteration_results": results,
                "first_reward":      first_reward,
                "final_reward":      last_reward,
                "improvement":       improvement,
                "metrics_path":      str(metrics_path),
                "memory_stats":      memory.section_stats(),
            }

        except Exception as e:
            raise DischargeAgentException(e, sys)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summary_to_sections(summary) -> dict[str, str]:
    """Flatten DischargeSummary fields into a {section: text} dict."""
    def _join(lst): return "\n".join(lst) if lst else ""

    return {
        "demographics":          f"Name: {summary.patient_name or ''} | DOB: {summary.date_of_birth or ''} | MRN: {summary.mrn or ''}",
        "admission_date":        summary.admission_date or "MISSING",
        "discharge_date":        summary.discharge_date or "MISSING",
        "principal_diagnosis":   summary.principal_diagnosis or "MISSING",
        "secondary_diagnoses":   _join(summary.secondary_diagnoses),
        "hospital_course":       summary.hospital_course or "MISSING",
        "procedures":            _join(summary.procedures),
        "admission_medications": _join(summary.admission_medications),
        "discharge_medications": _join(summary.discharge_medications),
        "allergies":             _join(summary.allergies),
        "follow_up":             _join(summary.follow_up_instructions),
        "pending_results":       _join([r.test_name for r in summary.pending_results]),
        "discharge_condition":   summary.discharge_condition or "MISSING",
    }


def _save_metrics(
    patient_id: str,
    results: list[IterationResult],
    metrics_dir: Path,
    held_out_sections: list[str],
) -> Path:
    """Save full metrics report as JSON and a readable text summary."""
    report = {
        "patient_id":        patient_id,
        "total_iterations":  len(results),
        "held_out_sections": held_out_sections,
        "improvement_curve": [
            {
                "iteration":       r.iteration,
                "avg_reward":      round(r.avg_reward, 4),
                "avg_edit_distance": round(r.avg_edit_distance, 4),
                "section_match_rate": round(r.section_match_rate, 4),
                "sections": [
                    {
                        "section":      s.section,
                        "reward":       s.reward,
                        "edit_distance":s.edit_distance,
                        "unchanged":    s.unchanged,
                    }
                    for s in r.section_scores
                ],
            }
            for r in results
        ],
        "before_after": {
            "first_avg_reward":   round(results[0].avg_reward,  4) if results else 0,
            "final_avg_reward":   round(results[-1].avg_reward, 4) if results else 0,
            "improvement":        round((results[-1].avg_reward - results[0].avg_reward), 4) if results else 0,
            "first_edit_distance":round(results[0].avg_edit_distance,  4) if results else 1,
            "final_edit_distance":round(results[-1].avg_edit_distance, 4) if results else 1,
        },
    }

    json_path = metrics_dir / f"metrics_{patient_id}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    txt_path = metrics_dir / f"metrics_{patient_id}.txt"
    lines = [
        f"Learning Loop Metrics — {patient_id}",
        "=" * 50,
        f"Iterations run:  {len(results)}",
        f"Held-out sections: {', '.join(held_out_sections)}",
        "",
        "Improvement Curve:",
        f"{'Iter':>5} {'Reward':>8} {'EditDist':>10} {'MatchRate':>10}",
        "-" * 38,
    ]
    for r in results:
        lines.append(
            f"{r.iteration:>5} {r.avg_reward:>8.3f} {r.avg_edit_distance:>10.3f} {r.section_match_rate:>9.1%}"
        )
    lines += [
        "",
        "Before / After:",
        f"  Reward:        {report['before_after']['first_avg_reward']:.3f} → {report['before_after']['final_avg_reward']:.3f}  (Δ{report['before_after']['improvement']:+.3f})",
        f"  Edit distance: {report['before_after']['first_edit_distance']:.3f} → {report['before_after']['final_edit_distance']:.3f}",
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Metrics saved → %s", json_path)
    return json_path
