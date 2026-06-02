"""
RAG pipeline: chunk → BM25 index → retrieve.
Pure-Python BM25 — no heavy ML deps required.
"""
from __future__ import annotations
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _chunk_text(
    text: str, source: str, page: int, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    chunks = []
    start, idx = 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(id=f"{source}:{page}:{idx}", source=source, page=page, text=chunk_text))
            idx += 1
        start = end - overlap
    return chunks


class BM25Index:
    """BM25 (Okapi BM25) retrieval over a corpus of string documents."""

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._tokenized: list[list[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0

    def fit(self, documents: list[str]) -> None:
        self._docs = documents
        self._tokenized = [_tokenize(d) for d in documents]
        self._df = Counter()
        for tokens in self._tokenized:
            for tok in set(tokens):
                self._df[tok] += 1
        self._avgdl = sum(len(t) for t in self._tokenized) / max(1, len(self._tokenized))

    def score(self, query: str, top_n: int = 6) -> list[tuple[int, float]]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        N = len(self._docs)
        scores = []
        for i, tokens in enumerate(self._tokenized):
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                freq = tf[tok]
                n_tok = self._df.get(tok, 0)
                idf = math.log((N - n_tok + 0.5) / (n_tok + 0.5) + 1)
                numerator = freq * (self.K1 + 1)
                denominator = freq + self.K1 * (1 - self.B + self.B * dl / max(1, self._avgdl))
                score += idf * numerator / denominator
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_n] if s > 0]


class RAGPipeline:
    """In-memory RAG using BM25 retrieval (zero heavy deps)."""

    def __init__(self, collection_name: str = "patient_docs") -> None:
        self._chunks: list[Chunk] = []
        self._index = BM25Index()

    def index_documents(self, docs: dict) -> int:
        all_chunks: list[Chunk] = []
        for doc_name, doc_content in docs.items():
            if doc_content.error:
                logger.warning(f"Skipping {doc_name}: {doc_content.error}")
                continue
            for page in doc_content.pages:
                if page.text.strip():
                    all_chunks.extend(_chunk_text(page.text, doc_name, page.page_num))

        if not all_chunks:
            logger.warning("No content to index")
            return 0

        self._chunks = all_chunks
        self._index.fit([c.text for c in all_chunks])
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents")
        return len(all_chunks)

    def query(self, question: str, n_results: int = 6) -> list[dict]:
        if not self._chunks:
            return []
        hits = self._index.score(question, top_n=n_results)
        return [
            {
                "text": self._chunks[i].text,
                "source": self._chunks[i].source,
                "page": self._chunks[i].page,
                "distance": round(1 / (1 + score), 4),
            }
            for i, score in hits
        ]

    def query_section(self, section: str, extra_context: str = "") -> str:
        queries = {
            "demographics": "patient name date of birth MRN medical record number age gender",
            "admission_date": "admission date admitted hospital",
            "discharge_date": "discharge date discharged",
            "principal_diagnosis": "principal diagnosis primary diagnosis main diagnosis",
            "secondary_diagnoses": "secondary diagnosis comorbidities other diagnoses",
            "hospital_course": "hospital course treatment clinical course summary events",
            "procedures": "procedures operations surgery interventions performed",
            "admission_medications": "admission medications home medications on admission",
            "discharge_medications": "discharge medications on discharge prescribed",
            "allergies": "allergies allergic reactions drug allergy NKDA no known",
            "follow_up": "follow up instructions outpatient clinic appointment referral",
            "pending_results": "pending results awaiting outstanding labs cultures",
            "discharge_condition": "discharge condition stable improved critical",
        }
        q = queries.get(section, section)
        if extra_context:
            q += f" {extra_context}"
        chunks = self.query(q, n_results=6)
        if not chunks:
            return "[MISSING — no relevant content found]"
        return "\n---\n".join(
            f"[Source: {c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks
        )
