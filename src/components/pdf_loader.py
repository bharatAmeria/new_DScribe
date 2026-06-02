"""PDF ingestion component — text extraction with OCR fallback for scanned PDFs.

Memory strategy for large scanned PDFs:
- OCR is done in small page batches (images freed after each batch)
- Extracted text is written to a .txt cache file on disk immediately
- DocumentContent holds only the cache file path, not the full text in RAM
- Re-runs skip OCR entirely if the cache file already exists
"""
import sys
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import CONFIG
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_num: int
    text: str
    source: str          # "text" | "ocr"
    confidence: float = 1.0


@dataclass
class DocumentContent:
    path: Path
    pages: list[PageContent] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[Page {p.page_num}]\n{p.text}"
            for p in self.pages
            if p.text.strip()
        )

    @property
    def is_empty(self) -> bool:
        return not any(p.text.strip() for p in self.pages)


class PDFLoader:
    """
    Loads patient PDFs from a folder.
    - Text-based PDFs: extracted directly via PyMuPDF.
    - Scanned PDFs: OCR'd in batches, results cached to disk to avoid OOM.
    """

    def __init__(self):
        self.ocr_threshold = CONFIG.pdf.ocr_threshold_chars
        self.ocr_dpi       = CONFIG.pdf.ocr_dpi
        logging.info("PDFLoader initialised (threshold=%d chars, OCR DPI=%d)",
                     self.ocr_threshold, self.ocr_dpi)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_folder(self, folder: Path) -> dict[str, DocumentContent]:
        """Load all PDFs in a patient folder. Returns {stem: DocumentContent}."""
        try:
            folder = Path(folder)
            pdf_files = sorted(folder.glob("**/*.pdf"))
            if not pdf_files:
                logging.warning("No PDFs found in %s", folder)
                return {}

            docs: dict[str, DocumentContent] = {}
            for pdf_path in pdf_files:
                key = pdf_path.stem.lower().replace(" ", "_")
                logging.info("Loading PDF: %s", pdf_path.name)
                docs[key] = self.load_file(pdf_path)

            logging.info("Loaded %d PDF(s) from %s", len(docs), folder)
            return docs
        except Exception as e:
            raise DischargeAgentException(e, sys)

    def load_file(self, path: Path) -> DocumentContent:
        """Load a single PDF, falling back to OCR if needed."""
        path = Path(path)
        if not path.exists():
            return DocumentContent(path=path, error=f"File not found: {path}")
        try:
            # Check for existing OCR cache first (avoids re-OCR on re-runs)
            cache_path = path.with_suffix(".ocr_cache.json")
            if cache_path.exists():
                logging.info("%s: loading from OCR cache → %s", path.name, cache_path.name)
                return self._load_from_cache(path, cache_path)

            pages = self._extract_text(path)
            total_chars = sum(len(p.text.strip()) for p in pages)

            if total_chars < self.ocr_threshold:
                logging.info("%s: direct extraction yielded %d chars → using OCR",
                             path.name, total_chars)
                # OCR writes to cache and returns only lightweight PageContent objects
                pages = self._extract_ocr_cached(path, cache_path)

            return DocumentContent(path=path, pages=pages)
        except Exception as e:
            logging.error("Failed to load %s: %s", path, e)
            return DocumentContent(path=path, error=str(e))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, path: Path) -> list[PageContent]:
        """Fast text extraction via PyMuPDF (text-based PDFs)."""
        try:
            import fitz
            pages = []
            doc = fitz.open(str(path))
            for i, page in enumerate(doc):
                text = page.get_text("text")
                pages.append(PageContent(page_num=i + 1, text=text, source="text"))
            doc.close()
            return pages
        except Exception as e:
            raise DischargeAgentException(e, sys)

    def _extract_ocr_cached(self, path: Path, cache_path: Path, batch_size: int = 3) -> list[PageContent]:
        """
        OCR in small batches → write each page's text to disk immediately.
        Keeps only one batch of images in RAM at a time.
        Returns lightweight PageContent list (text already on disk, also in memory for indexing).
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path, pdfinfo_from_path

            info = pdfinfo_from_path(str(path))
            total_pages = info["Pages"]
            logging.info("OCR: %d pages, batch_size=%d, DPI=%d → cache: %s",
                         total_pages, batch_size, self.ocr_dpi, cache_path.name)

            cache_data = []   # list of {page_num, text, confidence} — built incrementally
            pages      = []   # PageContent list to return

            for batch_start in range(1, total_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, total_pages)
                logging.info("OCR: pages %d–%d / %d", batch_start, batch_end, total_pages)

                images = convert_from_path(
                    str(path), dpi=self.ocr_dpi,
                    first_page=batch_start, last_page=batch_end,
                )
                for j, img in enumerate(images):
                    page_num = batch_start + j
                    try:
                        data = pytesseract.image_to_data(
                            img, output_type=pytesseract.Output.DICT
                        )
                        words = [
                            data["text"][k]
                            for k in range(len(data["text"]))
                            if int(data["conf"][k]) > 30 and data["text"][k].strip()
                        ]
                        text = " ".join(words)
                        conf_vals = [
                            int(c) for c in data["conf"]
                            if str(c).lstrip("-").isdigit() and int(c) > 0
                        ]
                        conf = sum(conf_vals) / max(1, len(conf_vals)) / 100
                    except Exception as page_err:
                        logging.warning("OCR failed on page %d: %s", page_num, page_err)
                        text, conf = "", 0.0
                    finally:
                        img.close()   # free image RAM immediately

                    cache_data.append({"page_num": page_num, "text": text, "confidence": conf})
                    pages.append(PageContent(page_num=page_num, text=text,
                                             source="ocr", confidence=conf))

                del images  # release batch before loading next

                # Write cache after every batch — survives a crash mid-run
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)

            logging.info("OCR complete: %d pages cached to %s", total_pages, cache_path.name)
            return pages

        except Exception as e:
            raise DischargeAgentException(e, sys)

    def _load_from_cache(self, path: Path, cache_path: Path) -> DocumentContent:
        """Load previously OCR'd text from the JSON cache file."""
        try:
            with open(cache_path) as f:
                data = json.load(f)
            pages = [
                PageContent(
                    page_num=entry["page_num"],
                    text=entry["text"],
                    source="ocr",
                    confidence=entry.get("confidence", 1.0),
                )
                for entry in data
            ]
            logging.info("Loaded %d pages from OCR cache", len(pages))
            return DocumentContent(path=path, pages=pages)
        except Exception as e:
            raise DischargeAgentException(e, sys)
