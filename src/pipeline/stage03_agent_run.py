import sys
import logging
from pathlib import Path
from typing import Optional

from src.constants import AGENT_RUN_STAGE_NAME
from src.agent.graph import run_agent
from src.components.rag_pipeline import RAGPipeline
from src.config import CONFIG
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


class AgentRunPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main(
        patient_id: str,
        patient_folder: Path,
        pre_indexed_rag: Optional[RAGPipeline] = None,
    ) -> dict:
        """
        Run the discharge summary agent.
        Pass pre_indexed_rag from Stage 2 to skip re-OCR entirely.
        """
        output_dir = Path(CONFIG.patients.output_dir)
        result = run_agent(
            patient_id=patient_id,
            patient_folder=patient_folder,
            output_dir=output_dir,
            max_iterations=CONFIG.agent.max_iterations,
            pre_indexed_rag=pre_indexed_rag,
        )
        summary = result.get("summary")
        logger.info(
            "Agent complete for '%s': %d flag(s), %d escalation(s), trace → %s",
            patient_id,
            len(summary.flags) if summary else 0,
            len(result["escalation_log"]),
            result["trace_path"],
        )
        return result
