"""
ChromaDB vector store with FastEmbed ONNX embeddings.

Design:
- Persistent: index survives restarts (no re-embedding on re-runs)
- Lightweight: FastEmbed uses ONNX runtime (~33MB model, no GPU/PyTorch needed)
- Per-patient collections: each patient gets an isolated ChromaDB collection
- Same query interface as the old BM25 pipeline
"""
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import CONFIG
from src.constants import SECTION_QUERIES
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def _chunk_text(
    text: str,
    source: str,
    page: int,
    size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text into overlapping chunks."""

    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size})"
        )

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    id=f"{source}:{page}:{idx}",
                    source=source,
                    page=page,
                    text=chunk_text,
                )
            )
            idx += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


class VectorStore:
    """
    ChromaDB-backed vector store with FastEmbed embeddings.

    Usage:
        store = VectorStore(patient_id="patient_1")
        store.index_documents(docs)                   # embed + persist
        results = store.query("discharge medications") # semantic search
    """

    def __init__(self, patient_id: str):
        try:
            cfg = self._cfg()
            self._patient_id    = patient_id
            self._collection_name = f"{cfg['prefix']}_{patient_id}"
            self._persist_dir   = Path(cfg["persist_dir"])
            self._model_name    = cfg["model"]
            self._n_results     = cfg["n_results"]
            self._chunk_size    = CONFIG.rag.chunk_size
            self._chunk_overlap = CONFIG.rag.chunk_overlap

            self._persist_dir.mkdir(parents=True, exist_ok=True)

            # Lazy-initialised to avoid import cost at module load
            self._client     = None
            self._collection = None
            self._embedder   = None

            logger.info(
                "VectorStore ready (patient=%s, model=%s, persist=%s)",
                patient_id, self._model_name, self._persist_dir,
            )
        except Exception as e:
            raise DischargeAgentException(e, sys)

    # ── Public API ────────────────────────────────────────────────────────────

    def index_documents(self, docs: dict) -> int:
        """
        Chunk all documents, embed, and upsert into ChromaDB.
        Memory-safe version using batched embedding.
        """

        try:
            all_chunks: list[Chunk] = []

            logger.info("Starting chunk generation")

            for doc_name, doc_content in docs.items():
                if doc_content.error:
                    logger.warning(
                        "Skipping %s: %s",
                        doc_name,
                        doc_content.error,
                    )
                    continue

                for page in doc_content.pages:
                    if page.text.strip():
                        all_chunks.extend(
                            _chunk_text(
                                page.text,
                                doc_name,
                                page.page_num,
                                self._chunk_size,
                                self._chunk_overlap,
                            )
                        )

            if not all_chunks:
                logger.warning("No content to index")
                return 0

            logger.info(
                "Generated %d chunks from documents",
                len(all_chunks),
            )

            self._init_store()

            existing = self._collection.count()

            if existing > 0:
                logger.info(
                    "Collection '%s' already contains %d chunks",
                    self._collection_name,
                    existing,
                )

            embed_batch_size = 32

            for start_idx in range(0, len(all_chunks), embed_batch_size):

                batch_chunks = all_chunks[
                    start_idx:start_idx + embed_batch_size
                ]

                texts = [c.text for c in batch_chunks]

                ids = [c.id for c in batch_chunks]

                metadatas = [
                    {
                        "source": c.source,
                        "page": c.page,
                    }
                    for c in batch_chunks
                ]

                logger.info(
                    "Embedding batch %d-%d/%d",
                    start_idx + 1,
                    min(
                        start_idx + embed_batch_size,
                        len(all_chunks),
                    ),
                    len(all_chunks),
                )

                embeddings = self._embed(texts)

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

            logger.info(
                "Indexed %d chunks into collection '%s'",
                len(all_chunks),
                self._collection_name,
            )

            return len(all_chunks)

        except Exception as e:
            raise DischargeAgentException(e, sys)

    def query(self, question: str, n_results: int | None = None) -> list[dict]:
        """
        Semantic search: returns top-n chunks most relevant to question.
        Each result: {text, source, page, distance}
        """
        try:
            self._init_store()
            n = n_results or self._n_results
            count = self._collection.count()
            if count == 0:
                return []

            q_embedding = self._embed([question])[0]
            results = self._collection.query(
                query_embeddings=[q_embedding],
                n_results=min(n, count),
                include=["documents", "metadatas", "distances"],
            )

            output = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                output.append({
                    "text":     doc,
                    "source":   meta.get("source", "unknown"),
                    "page":     meta.get("page", 0),
                    "distance": round(dist, 4),
                })
            return output

        except Exception as e:
            raise DischargeAgentException(e, sys)

    def query_section(self, section: str) -> str:
        """
        Retrieve context for a named discharge summary section.
        Returns concatenated chunks or a MISSING marker.
        """
        q = SECTION_QUERIES.get(section, section)
        chunks = self.query(q)
        if not chunks:
            return "[MISSING — no relevant content found]"
        return "\n---\n".join(
            f"[Source: {c['source']}, p.{c['page']}]\n{c['text']}"
            for c in chunks
        )

    def delete_collection(self) -> None:
        """Drop the patient's collection (useful for re-indexing from scratch)."""
        try:
            self._init_client()
            self._client.delete_collection(self._collection_name)
            logger.info("Deleted collection '%s'", self._collection_name)
        except Exception as e:
            raise DischargeAgentException(e, sys)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_client(self) -> None:
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))

    def _init_store(self) -> None:
        """Lazy init of ChromaDB client, collection, and embedder."""
        if self._collection is not None:
            return
        self._init_client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if self._embedder is None:
            from fastembed import TextEmbedding
            logger.info("Loading embedding model: %s", self._model_name)
            self._embedder = TextEmbedding(model_name=self._model_name)
            logger.info("Embedding model loaded")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

        logger.debug(
            "Embedding %d chunks",
            len(texts),
        )

        vectors = []

        for vector in self._embedder.embed(texts):
            vectors.append(vector.tolist())

        return vectors
        
    @staticmethod
    def _cfg() -> dict:
        vs = CONFIG.vectorstore
        return {
            "persist_dir": vs.persist_dir,
            "model":       vs.embedding_model,
            "n_results":   vs.n_results,
            "prefix":      vs.collection_prefix,
        }
