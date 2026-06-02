import sys
import logging

from src.constants import RAG_INDEXING_STAGE_NAME
from src.components.rag_pipeline import RAGPipeline
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


class RAGIndexingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main(docs: dict, patient_id: str = "default") -> RAGPipeline:
        """
        Chunk, embed, and index loaded documents.
        Uses ChromaDB vector store (FastEmbed) with BM25 fallback.
        Returns the initialised RAGPipeline.
        """
        rag      = RAGPipeline(patient_id=patient_id)
        n_chunks = rag.index_documents(docs)
        logger.info(
            "RAG indexing complete: %d chunks indexed [backend: %s]",
            n_chunks, rag.backend,
        )
        return rag


if __name__ == "__main__":
    try:
        logger.info(">>>>>> stage %s started <<<<<<", RAG_INDEXING_STAGE_NAME)
        from pathlib import Path
        from src.config import CONFIG
        from src.pipeline.stage01_pdf_ingestion import PDFIngestionPipeline

        folder = Path(CONFIG.patients.base_dir) / "patient_1"
        docs   = PDFIngestionPipeline.main(folder)
        RAGIndexingPipeline.main(docs, patient_id="patient_1")
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", RAG_INDEXING_STAGE_NAME)
    except DischargeAgentException as e:
        raise DischargeAgentException(e, sys)
