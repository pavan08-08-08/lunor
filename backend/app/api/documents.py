import logging
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import UPLOADS_DIR, VECTORSTORE_DIR, ensure_directories
from app.rag.bm25_store import build_bm25_index, save_bm25_index
from app.rag.chunker import ChunkDocument, chunk_pages
from app.rag.embeddings import embed_chunks
from app.rag.loader import PDFLoadError, load_pdf
from app.rag.vector_store import build_vector_store, save_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentItem(BaseModel):
    filename: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int


def rebuild_vector_store() -> None:
    """Find all PDFs in data/uploads, chunk and embed them, and rebuild the FAISS store."""
    ensure_directories()
    pdf_files = sorted(UPLOADS_DIR.glob("*.pdf"))
    if not pdf_files:
        return

    all_chunks: list[ChunkDocument] = []
    for pdf_path in pdf_files:
        pages = load_pdf(str(pdf_path))
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        return

    embedded_chunks = embed_chunks(all_chunks)
    store = build_vector_store(embedded_chunks)
    save_vector_store(store, str(VECTORSTORE_DIR))

    bm25_index = build_bm25_index(all_chunks)
    save_bm25_index(bm25_index, str(VECTORSTORE_DIR))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a PDF document, validate it, save it, and rebuild the vector store."""
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    original_name = file.filename.strip()
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read content to validate PDF magic signature
    content = await file.read()
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file format")

    # Prevent path traversal by extracting strictly the basename
    safe_filename = Path(original_name).name
    ensure_directories()
    target_path = (UPLOADS_DIR / safe_filename).resolve()

    # Verify target path remains within UPLOADS_DIR
    if target_path.parent != UPLOADS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        target_path.write_bytes(content)
    except Exception as exc:
        logger.error("Failed to write uploaded file to disk: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    try:
        # Load and chunk the uploaded PDF to get page and chunk counts
        pages = load_pdf(str(target_path))
        chunks = chunk_pages(pages)

        # Rebuild the FAISS index including all uploaded PDFs
        rebuild_vector_store()

        return DocumentUploadResponse(
            message="Document uploaded successfully",
            filename=safe_filename,
            pages=len(pages),
            chunks=len(chunks),
        )
    except PDFLoadError as exc:
        # Clean up file on extraction failure
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(status_code=400, detail=f"PDF extraction failure: {exc}")
    except Exception as exc:
        logger.error("Indexing failed for %s: %s", safe_filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process and index document")


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List all uploaded PDF documents currently stored in data/uploads."""
    ensure_directories()
    if not UPLOADS_DIR.exists():
        return DocumentListResponse(documents=[])

    pdf_files = sorted(UPLOADS_DIR.glob("*.pdf"), key=lambda p: p.name)
    return DocumentListResponse(
        documents=[DocumentItem(filename=pdf.name) for pdf in pdf_files]
    )
