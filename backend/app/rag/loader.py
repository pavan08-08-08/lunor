from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import pymupdf

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when loading a PDF fails at the file level."""
    pass


@dataclass
class PageDocument:
    """Represents a single page extracted from a PDF document."""
    source_filename: str
    page_number: int
    total_pages: int
    text: str
    char_count: int
    is_empty: bool
    doc_id: str


def _generate_doc_id(file_path: Path) -> str:
    """Generate a stable, content-based document ID using SHA-256."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: str) -> list[PageDocument]:
    """Load a PDF file and extract text page-by-page into PageDocument instances.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of PageDocument instances, one per page in the document.

    Raises:
        PDFLoadError: If the file does not exist, is not a regular file, is corrupted/invalid,
            or is encrypted/password-protected.
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFLoadError(f"PDF file not found: '{file_path}'")

    if not path.is_file():
        raise PDFLoadError(f"Path is not a regular file: '{file_path}'")

    try:
        doc_id = _generate_doc_id(path)
    except Exception as exc:
        raise PDFLoadError(f"Failed to read file '{file_path}': {exc}") from exc

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise PDFLoadError(f"Failed to open PDF '{file_path}': {exc}") from exc

    pages: list[PageDocument] = []
    source_filename = path.name

    with doc:
        if doc.is_encrypted or doc.needs_pass:
            raise PDFLoadError(f"PDF file is encrypted or password-protected: '{file_path}'")

        total_pages = doc.page_count

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            try:
                page = doc.load_page(page_idx)
                page_text = page.get_text() or ""
            except Exception as exc:
                logger.error(
                    "Failed to extract text from page %d of '%s': %s",
                    page_number,
                    source_filename,
                    exc,
                    exc_info=True,
                )
                page_text = ""

            is_empty = not page_text.strip()
            pages.append(
                PageDocument(
                    source_filename=source_filename,
                    page_number=page_number,
                    total_pages=total_pages,
                    text=page_text,
                    char_count=len(page_text),
                    is_empty=is_empty,
                    doc_id=doc_id,
                )
            )

    return pages
