import sys
import logging
from pathlib import Path

from src.constants import PDF_INGESTION_STAGE_NAME
from src.components.pdf_loader import PDFLoader
from src.config import CONFIG
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


class PDFIngestionPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main(patient_folder: Path) -> dict:
        """Load all PDFs for a patient and return {name: DocumentContent}."""
        loader = PDFLoader()
        docs   = loader.load_folder(patient_folder)
        logger.info(
            "PDF ingestion complete: %d document(s) loaded from %s",
            len(docs), patient_folder,
        )
        return docs


if __name__ == "__main__":
    try:
        logger.info(">>>>>> stage %s started <<<<<<", PDF_INGESTION_STAGE_NAME)
        folder = Path(CONFIG.patients.base_dir) / "patient_1"
        result = PDFIngestionPipeline.main(folder)
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", PDF_INGESTION_STAGE_NAME)
    except DischargeAgentException as e:
        raise DischargeAgentException(e, sys)
