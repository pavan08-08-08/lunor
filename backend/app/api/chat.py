import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.config import VECTORSTORE_DIR
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve
from app.rag.vector_store import load_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SourceCitationModel(BaseModel):
    source_filename: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitationModel]
    has_sufficient_context: bool


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a user question using grounded retrieval and Gemini generation."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Verify that the vector store exists
    index_file = VECTORSTORE_DIR / "index.faiss"
    pkl_file = VECTORSTORE_DIR / "index.pkl"
    if not (index_file.exists() and pkl_file.exists()):
        raise HTTPException(
            status_code=400,
            detail="No documents are indexed. Upload a PDF before asking questions.",
        )

    try:
        store = load_vector_store(str(VECTORSTORE_DIR))
    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load vector store")

    try:
        retrieved_chunks = retrieve(query=query, store=store)
    except Exception as exc:
        logger.error("Retrieval failed for query '%s': %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant context")

    try:
        generated = generate_answer(query=query, retrieved_chunks=retrieved_chunks)
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    return ChatResponse(
        answer=generated.answer,
        sources=[
            SourceCitationModel(
                source_filename=s.source_filename,
                page_number=s.page_number,
            )
            for s in generated.sources
        ],
        has_sufficient_context=generated.has_sufficient_context,
    )
