"""
Main entry point — runs all three pipeline stages for every patient.
RAG built in Stage 2 is passed directly into Stage 3 → no re-OCR.
"""
import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.logger import configure_logger          # noqa: F401
from src.constants import (
    PDF_INGESTION_STAGE_NAME,
    RAG_INDEXING_STAGE_NAME,
    AGENT_RUN_STAGE_NAME,
)
from src.config import CONFIG
from src.exception import DischargeAgentException
from src.pipeline.stage01_pdf_ingestion import PDFIngestionPipeline
from src.pipeline.stage02_rag_indexing import RAGIndexingPipeline
from src.pipeline.stage03_agent_run import AgentRunPipeline
from src.pipeline.stage04_learning_loop import LearningLoopPipeline

logger = logging.getLogger(__name__)


def run_patient(patient_id: str, patient_folder: Path):
    """Run all three stages. Returns the indexed RAG (reused by learning loop)."""
    rag = None

    # ── Stage 1: PDF Ingestion ────────────────────────────────────────────────
    try:
        logger.info(">>>>>> stage %s started for %s <<<<<<",
                    PDF_INGESTION_STAGE_NAME, patient_id)
        docs = PDFIngestionPipeline.main(patient_folder)
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x",
                    PDF_INGESTION_STAGE_NAME)
    except DischargeAgentException as e:
        logger.exception(e)
        raise e

    # ── Stage 2: RAG Indexing ─────────────────────────────────────────────────
    try:
        logger.info(">>>>>> stage %s started for %s <<<<<<",
                    RAG_INDEXING_STAGE_NAME, patient_id)
        rag = RAGIndexingPipeline.main(docs, patient_id=patient_id)
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x",
                    RAG_INDEXING_STAGE_NAME)
    except DischargeAgentException as e:
        logger.exception(e)
        raise e

    # ── Stage 3: Agent Run (reuses RAG from Stage 2 — no re-OCR) ─────────────
    try:
        logger.info(">>>>>> stage %s started for %s <<<<<<",
                    AGENT_RUN_STAGE_NAME, patient_id)
        result = AgentRunPipeline.main(
            patient_id=patient_id,
            patient_folder=patient_folder,
            pre_indexed_rag=rag,          # ← key: skip re-OCR
        )
        summary = result.get("summary")
        if summary:
            logger.info(
                "Patient %s | Dx: %s | Flags: %d | Missing: %s | Escalations: %d",
                patient_id,
                summary.principal_diagnosis or "MISSING",
                len(summary.flags),
                summary.missing_fields,
                len(result["escalation_log"]),
            )
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x",
                    AGENT_RUN_STAGE_NAME)
    except DischargeAgentException as e:
        logger.exception(e)
        raise e

    return rag   # caller can pass this to the learning loop


def run_learning_loop(patient_id: str, patient_folder: Path, rag) -> None:
    """Run Part 2 learning loop for a patient (requires RAG from Stage 2)."""
    try:
        logger.info(">>>>>> stage %s started for %s <<<<<<",
                    LEARNING_LOOP_STAGE_NAME, patient_id)
        result = LearningLoopPipeline.main(
            patient_id=patient_id,
            patient_folder=patient_folder,
            pre_indexed_rag=rag,
        )
        logger.info(
            "Learning loop | reward: %.3f → %.3f (Δ%.3f) | metrics → %s",
            result["first_reward"], result["final_reward"],
            result["improvement"], result["metrics_path"],
        )
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x",
                    LEARNING_LOOP_STAGE_NAME)
    except DischargeAgentException as e:
        logger.exception(e)
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Discharge Summary Agent")
    parser.add_argument("--patient",  type=str, default=None,
                        help="Patient subfolder name. Omit to run all patients.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base artifacts directory (default: from config.yaml)")
    parser.add_argument("--learn",    action="store_true",
                        help="Run Part 2 learning loop after Part 1 (requires --patient)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of learning iterations (default: from config.yaml)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir or CONFIG.patients.base_dir)

    if args.patient:
        folder = base_dir / args.patient
        if not folder.exists():
            logger.error("Patient folder not found: %s", folder)
            sys.exit(1)
        rag = run_patient(args.patient, folder)
        if args.learn:
            if rag is None:
                logger.error("--learn requires a successful Part 1 run first")
                sys.exit(1)
            run_learning_loop(args.patient, folder, rag)
    else:
        patient_folders = [d for d in sorted(base_dir.iterdir())
                           if d.is_dir() and d.name.startswith("patient")]
        if not patient_folders:
            logger.error("No patient folders found under %s", base_dir)
            sys.exit(1)
        logger.info("Found %d patient(s): %s", len(patient_folders),
                    [d.name for d in patient_folders])
        for folder in patient_folders:
            logger.info("=" * 60)
            logger.info("Processing: %s", folder.name)
            logger.info("=" * 60)
            try:
                run_patient(folder.name, folder)
            except Exception as e:
                logger.error("Failed for %s: %s", folder.name, e)


if __name__ == "__main__":
    main()
