#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discharge_agent.pdf_loader import DocumentContent, PageContent
from discharge_agent.rag import RAGPipeline

doc = DocumentContent(
    path=Path("n.pdf"),
    pages=[PageContent(1, "PATIENT Jane Smith MRN 87654321 diagnosis pneumonia medications furosemide", "text")]
)
rag = RAGPipeline("t")
n = rag.index_documents({"note": doc})
print(f"indexed {n} chunks")
res = rag.query("patient MRN diagnosis", n_results=3)
print(f"query returned {len(res)} results")
assert "Jane Smith" in res[0]["text"]
print("RAG test PASSED")
