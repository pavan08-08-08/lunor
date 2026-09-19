import logging
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
import pymupdf
from pydantic import BaseModel

from app.config import UPLOADS_DIR, VECTORSTORE_DIR, ensure_directories
from app.rag.bm25_store import build_bm25_index, load_bm25_index, save_bm25_index
from app.rag.chunker import ChunkDocument, chunk_pages
from app.rag.embeddings import embed_chunks
from app.rag.evidence import extract_evidence_passage, locate_evidence_rectangles
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


class DocumentDeleteResponse(BaseModel):
    message: str
    filename: str
    remaining_documents: int


class HighlightRect(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PageRenderResponse(BaseModel):
    filename: str
    page_number: int
    total_pages: int
    page_width: float
    page_height: float
    svg: str
    evidence_text: str
    highlights: list[HighlightRect]


def _cleanup_vector_store_files() -> None:
    """Delete FAISS and BM25 index files from VECTORSTORE_DIR when no indexed chunks remain."""
    for filename in ["index.faiss", "index.pkl", "bm25.pkl"]:
        index_path = VECTORSTORE_DIR / filename
        if index_path.exists():
            try:
                index_path.unlink()
            except OSError as exc:
                logger.warning("Failed to delete stale index file %s: %s", index_path, exc)


def rebuild_vector_store() -> None:
    """Find all PDFs in data/uploads, chunk and embed them, and rebuild the FAISS store."""
    ensure_directories()
    pdf_files = sorted(UPLOADS_DIR.glob("*.pdf"))
    if not pdf_files:
        _cleanup_vector_store_files()
        return

    all_chunks: list[ChunkDocument] = []
    for pdf_path in pdf_files:
        pages = load_pdf(str(pdf_path))
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        _cleanup_vector_store_files()
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


@router.delete("/{filename}", response_model=DocumentDeleteResponse)
def delete_document(filename: str) -> DocumentDeleteResponse:
    """Delete an uploaded PDF document, clean up storage, and rebuild search indices."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    raw_name = filename.strip()
    safe_filename = Path(raw_name).name

    # Reject directory traversal attempts or invalid filenames
    if safe_filename != raw_name or "/" in raw_name or "\\" in raw_name or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be deleted")

    ensure_directories()
    target_path = (UPLOADS_DIR / safe_filename).resolve()

    # Verify target path remains within UPLOADS_DIR
    if target_path.parent != UPLOADS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        target_path.unlink()
    except Exception as exc:
        logger.error("Failed to delete file %s from disk: %s", safe_filename, exc)
        raise HTTPException(status_code=500, detail="Failed to delete document from storage")

    try:
        rebuild_vector_store()
    except Exception as exc:
        logger.error("Failed to rebuild search indices after deleting %s: %s", safe_filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to rebuild search indices after deletion")

    remaining_files = sorted(UPLOADS_DIR.glob("*.pdf")) if UPLOADS_DIR.exists() else []
    return DocumentDeleteResponse(
        message="Document deleted successfully",
        filename=safe_filename,
        remaining_documents=len(remaining_files),
    )


@router.get("/{filename}/page", response_model=PageRenderResponse)
def get_document_page(
    filename: str,
    page_number: int = 1,
    chunk_id: str | None = None,
    query: str | None = None,
    evidence_text: str | None = None,
) -> PageRenderResponse:
    """Render a single page of a PDF as crisp SVG with backend-drawn evidence highlight."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    raw_name = filename.strip()
    safe_filename = Path(raw_name).name

    # Reject directory traversal attempts or invalid filenames
    if safe_filename != raw_name or "/" in raw_name or "\\" in raw_name or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be accessed")

    ensure_directories()
    target_path = (UPLOADS_DIR / safe_filename).resolve()

    # Verify target path remains within UPLOADS_DIR
    if target_path.parent != UPLOADS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    resolved_evidence = (evidence_text or "").strip()
    if chunk_id is not None:
        chunk_id = chunk_id.strip()
        if not chunk_id:
            raise HTTPException(status_code=400, detail="Chunk ID cannot be empty")

        bm25_file = VECTORSTORE_DIR / "bm25.pkl"
        matching_chunk = None
        if bm25_file.exists():
            try:
                bm25_index = load_bm25_index(str(VECTORSTORE_DIR))
                matching_chunk = next(
                    (c for c in bm25_index.chunks if c.get("chunk_id") == chunk_id),
                    None,
                )
            except Exception as exc:
                logger.warning("Could not load BM25 index to validate chunk %s: %s", chunk_id, exc)

        if not matching_chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        if matching_chunk.get("source_filename") != safe_filename:
            raise HTTPException(
                status_code=400,
                detail="Chunk does not belong to the requested document",
            )

        if not resolved_evidence:
            chunk_text = matching_chunk.get("text", "")
            resolved_evidence = extract_evidence_passage(chunk_text, query=query or "")

    try:
        raw_bytes = target_path.read_bytes()
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", safe_filename, exc)
        raise HTTPException(status_code=500, detail="Failed to read document")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise HTTPException(status_code=400, detail="Document contains no pages")

    if page_number < 1 or page_number > total_pages:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid page number {page_number}. Document has {total_pages} page(s).",
        )

    page_idx = page_number - 1
    page_obj = doc[page_idx]
    page_width = round(float(page_obj.rect.width), 2)
    page_height = round(float(page_obj.rect.height), 2)

    highlights: list[HighlightRect] = []
    if resolved_evidence:
        rect_dicts = locate_evidence_rectangles(page_obj, resolved_evidence)
        highlights = [HighlightRect(**r) for r in rect_dicts]

        # Draw translucent highlight rectangle directly onto the PyMuPDF page in memory
        for r in rect_dicts:
            page_obj.draw_rect(
                pymupdf.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]),
                color=(1.0, 0.65, 0.0),
                fill=(1.0, 0.65, 0.0),
                fill_opacity=0.30,
                overlay=False,
            )

    # Generate SVG after drawing the highlight so vector text and highlight are perfectly aligned
    svg_content = page_obj.get_svg_image()
    doc.close()

    return PageRenderResponse(
        filename=safe_filename,
        page_number=page_number,
        total_pages=total_pages,
        page_width=page_width,
        page_height=page_height,
        svg=svg_content,
        evidence_text=resolved_evidence,
        highlights=highlights,
    )


@router.get("/{filename}/file")
def get_document_file(
    filename: str,
    page: int | None = None,
    chunk_id: str | None = None,
    query: str | None = None,
) -> Response:
    """Serve the raw or in-memory highlighted PDF file for citation evidence viewing."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    raw_name = filename.strip()
    safe_filename = Path(raw_name).name

    # Reject directory traversal attempts or invalid filenames
    if safe_filename != raw_name or "/" in raw_name or "\\" in raw_name or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be accessed")

    ensure_directories()
    target_path = (UPLOADS_DIR / safe_filename).resolve()

    # Verify target path remains within UPLOADS_DIR
    if target_path.parent != UPLOADS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    if chunk_id is not None:
        chunk_id = chunk_id.strip()
        if not chunk_id:
            raise HTTPException(status_code=400, detail="Chunk ID cannot be empty")

        bm25_file = VECTORSTORE_DIR / "bm25.pkl"
        matching_chunk = None
        if bm25_file.exists():
            try:
                bm25_index = load_bm25_index(str(VECTORSTORE_DIR))
                matching_chunk = next(
                    (c for c in bm25_index.chunks if c.get("chunk_id") == chunk_id),
                    None,
                )
            except Exception as exc:
                logger.warning("Could not load BM25 index to validate chunk %s: %s", chunk_id, exc)

        if not matching_chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        if matching_chunk.get("source_filename") != safe_filename:
            raise HTTPException(
                status_code=400,
                detail="Chunk does not belong to the requested document",
            )

        target_page = matching_chunk.get("page_number", 1)
        chunk_text = matching_chunk.get("text", "")
        evidence_text = extract_evidence_passage(chunk_text, query=query or "")

        try:
            # Highlight on in-memory copy without altering disk file
            raw_bytes = target_path.read_bytes()
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
            page_idx = target_page - 1
            if 0 <= page_idx < len(doc):
                page_obj = doc[page_idx]
                rect_dicts = locate_evidence_rectangles(page_obj, evidence_text) if evidence_text else []
                if rect_dicts:
                    rects = [
                        pymupdf.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
                        for r in rect_dicts
                    ]
                    annot = page_obj.add_highlight_annot(rects)
                    annot.set_colors(stroke=(1.0, 0.9, 0.2))  # warm yellow
                    annot.update()

            annotated_bytes = doc.tobytes(garbage=3, deflate=True)
            doc.close()

            return Response(
                content=annotated_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'inline; filename="{safe_filename}"',
                },
            )
        except Exception as exc:
            logger.error("Failed to generate in-memory highlighted PDF: %s", exc, exc_info=True)
            # Fall back to serving original unannotated PDF rather than failing
            return FileResponse(
                path=target_path,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'inline; filename="{safe_filename}"',
                },
            )

    return FileResponse(
        path=target_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
        },
    )
