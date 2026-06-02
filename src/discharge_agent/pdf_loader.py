"""PDF ingestion with text extraction and OCR fallback for scanned documents."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_num: int
    text: str
    source: str  # "text" | "ocr"
    confidence: float = 1.0


@dataclass
class DocumentContent:
    path: Path
    pages: list[PageContent] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[Page {p.page_num}]\n{p.text}" for p in self.pages if p.text.strip()
        )

    @property
    def is_empty(self) -> bool:
        return not any(p.text.strip() for p in self.pages)


def _extract_text_pymupdf(path: Path) -> list[PageContent]:
    """Extract text using PyMuPDF (fast, works for text-based PDFs)."""
    import fitz  # PyMuPDF

    pages = []
    doc = fitz.open(str(path))
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(PageContent(page_num=i + 1, text=text, source="text"))
    doc.close()
    return pages


def _extract_text_ocr(path: Path, dpi: int = 200) -> list[PageContent]:
    """OCR fallback using pytesseract for scanned/image PDFs."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"OCR dependencies missing: {e}. Run: uv add pytesseract pdf2image pillow")

    pages = []
    images = convert_from_path(str(path), dpi=dpi)
    for i, img in enumerate(images):
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            # Filter low-confidence words
            words = [
                data["text"][j]
                for j in range(len(data["text"]))
                if int(data["conf"][j]) > 30 and data["text"][j].strip()
            ]
            text = " ".join(words)
            conf = sum(
                int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0
            ) / max(1, sum(1 for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0))
            pages.append(PageContent(page_num=i + 1, text=text, source="ocr", confidence=conf / 100))
        except Exception as e:
            logger.warning(f"OCR failed on page {i+1}: {e}")
            pages.append(PageContent(page_num=i + 1, text="", source="ocr", confidence=0.0))
    return pages


def load_pdf(path: Path, ocr_threshold_chars: int = 100) -> DocumentContent:
    """
    Load a PDF, auto-detecting whether OCR is needed.
    Falls back to OCR if extracted text is below threshold.
    """
    path = Path(path)
    if not path.exists():
        return DocumentContent(path=path, error=f"File not found: {path}")

    try:
        pages = _extract_text_pymupdf(path)
        total_chars = sum(len(p.text.strip()) for p in pages)

        if total_chars < ocr_threshold_chars:
            logger.info(f"{path.name}: text extraction yielded {total_chars} chars → using OCR")
            pages = _extract_text_ocr(path)

        return DocumentContent(path=path, pages=pages)

    except Exception as e:
        logger.error(f"PDF load failed for {path}: {e}")
        return DocumentContent(path=path, error=str(e))


def load_patient_folder(folder: Path) -> dict[str, DocumentContent]:
    """Load all PDFs in a patient folder, keyed by filename stem."""
    folder = Path(folder)
    docs: dict[str, DocumentContent] = {}
    pdf_files = sorted(folder.glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {folder}")

    for pdf_path in pdf_files:
        key = pdf_path.stem.lower().replace(" ", "_")
        logger.info(f"Loading: {pdf_path.name}")
        docs[key] = load_pdf(pdf_path)

    return docs
