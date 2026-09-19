from dataclasses import dataclass
from app.rag.loader import PageDocument

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


@dataclass
class ChunkDocument:
    """Represents a text chunk extracted from a PageDocument."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    text: str
    char_count: int


def chunk_pages(pages: list[PageDocument]) -> list[ChunkDocument]:
    """Chunk a list of PageDocument objects into ChunkDocument objects.

    Each page is chunked independently without cross-page concatenation.
    Empty and whitespace-only pages are skipped.

    Args:
        pages: List of PageDocument objects extracted from a document.

    Returns:
        A list of ChunkDocument objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[ChunkDocument] = []

    for page in pages:
        # Skip empty pages and defensively skip whitespace-only text
        if page.is_empty or not page.text or not page.text.strip():
            continue

        raw_chunks = splitter.split_text(page.text)
        chunk_index = 0

        for chunk_text in raw_chunks:
            if not chunk_text or not chunk_text.strip():
                continue

            chunk_id = f"{page.doc_id}_p{page.page_number}_c{chunk_index}"

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_count=len(chunk_text),
                )
            )
            chunk_index += 1

    return chunks
