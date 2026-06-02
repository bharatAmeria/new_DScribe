"""
RAG pipeline — uses ChromaDB vector store with FastEmbed embeddings.
Falls back to BM25 if the vector store is unavailable.
"""
import sys
import math
import logging
import re
from collections import Counter
from typing import Optional

from src.config import CONFIG
from src.constants import SECTION_QUERIES
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


# ── BM25 fallback (pure Python, zero deps) ───────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class _BM25Fallback:
    K1, B = 1.5, 0.75

    def __init__(self):
        self._docs: list[str] = []
        self._meta: list[dict] = []
        self._tokenized: list[list[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0

    def fit(self, texts: list[str], metadatas: list[dict]) -> None:
        self._docs = texts
        self._meta = metadatas
        self._tokenized = [_tokenize(t) for t in texts]
        self._df = Counter()
        for toks in self._tokenized:
            for tok in set(toks):
                self._df[tok] += 1
        self._avgdl = sum(len(t) for t in self._tokenized) / max(1, len(self._tokenized))

    def query(self, question: str, n: int = 6) -> list[dict]:
        if not self._docs:
            return []
        q_toks = _tokenize(question)
        N = len(self._docs)
        scores = []
        for i, toks in enumerate(self._tokenized):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for tok in q_toks:
                if tok not in tf:
                    continue
                freq = tf[tok]
                n_tok = self._df.get(tok, 0)
                idf = math.log((N - n_tok + 0.5) / (n_tok + 0.5) + 1)
                score += idf * freq * (self.K1 + 1) / (
                    freq + self.K1 * (1 - self.B + self.B * dl / max(1, self._avgdl))
                )
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "text":     self._docs[i],
                "source":   self._meta[i].get("source", "unknown"),
                "page":     self._meta[i].get("page", 0),
                "distance": round(1 / (1 + s), 4),
            }
            for i, s in scores[:n]
            if s > 0
        ]


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    RAG pipeline backed by ChromaDB vector store (FastEmbed ONNX embeddings).
    Automatically falls back to BM25 if ChromaDB/FastEmbed is unavailable.

    - Persistent: re-runs skip re-embedding if the collection already exists.
    - Per-patient: each patient_id gets its own isolated collection.
    """

    def __init__(self, patient_id: str = "default"):
        self._patient_id = patient_id
        self._store: Optional[object] = None   # VectorStore or _BM25Fallback
        self._using_vector_store = False
        logger.info(
            "RAGPipeline initialised (chunk_size=%d, overlap=%d)",
            CONFIG.rag.chunk_size, CONFIG.rag.chunk_overlap,
        )

    def index_documents(self, docs: dict) -> int:
        """Chunk, embed, and index all documents. Returns total chunks indexed."""
        try:
            # Try vector store first
            try:
                from src.components.vector_store import VectorStore
                self._store = VectorStore(patient_id=self._patient_id)
                n = self._store.index_documents(docs)
                self._using_vector_store = True
                logger.info("Using ChromaDB vector store (%d chunks)", n)
                return n
            except Exception as vs_err:
                logger.warning(
                    "VectorStore unavailable (%s) — falling back to BM25", vs_err
                )

            # BM25 fallback
            from src.components.vector_store import _chunk_text
            all_chunks = []
            for doc_name, doc_content in docs.items():
                if doc_content.error:
                    continue
                for page in doc_content.pages:
                    if page.text.strip():
                        all_chunks.extend(
                            _chunk_text(
                                page.text, doc_name, page.page_num,
                                CONFIG.rag.chunk_size, CONFIG.rag.chunk_overlap,
                            )
                        )
            if not all_chunks:
                logger.warning("No content to index")
                return 0

            bm25 = _BM25Fallback()
            bm25.fit(
                [c.text for c in all_chunks],
                [{"source": c.source, "page": c.page} for c in all_chunks],
            )
            self._store = bm25
            self._using_vector_store = False
            logger.info("Using BM25 fallback (%d chunks)", len(all_chunks))
            return len(all_chunks)

        except Exception as e:
            raise DischargeAgentException(e, sys)

    def query(self, question: str, n_results: int = 6) -> list[dict]:
        """Semantic (or BM25) search. Returns [{text, source, page, distance}]."""
        try:
            if self._store is None:
                return []
            if self._using_vector_store:
                return self._store.query(question, n_results=n_results)
            return self._store.query(question, n=n_results)
        except Exception as e:
            raise DischargeAgentException(e, sys)

    def query_section(self, section: str) -> str:
        """Retrieve context for a named discharge summary section."""
        q = SECTION_QUERIES.get(section, section)
        chunks = self.query(q)
        if not chunks:
            return "[MISSING — no relevant content found]"
        return "\n---\n".join(
            f"[Source: {c['source']}, p.{c['page']}]\n{c['text']}"
            for c in chunks
        )

    @property
    def backend(self) -> str:
        return "chromadb+fastembed" if self._using_vector_store else "bm25"
