from dataclasses import dataclass
from app.rag.chunker import ChunkDocument

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

_model: HuggingFaceEmbeddings | None = None


@dataclass
class EmbeddedChunk:
    """Represents a text chunk with its vector embedding and associated metadata."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    text: str
    char_count: int
    embedding: list[float]


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the lazy singleton HuggingFaceEmbeddings model instance on CPU."""
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _model


def embed_chunks(chunks: list[ChunkDocument]) -> list[EmbeddedChunk]:
    """Generate dense vector embeddings for a list of ChunkDocument objects.

    Calls the embedding model's batch embed_documents() method once and
    attaches the resulting vector to each chunk while preserving all metadata.

    Args:
        chunks: List of ChunkDocument objects to embed.

    Returns:
        List of EmbeddedChunk objects in matching order.
    """
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = model.embed_documents(texts)

    return [
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source_filename=chunk.source_filename,
            page_number=chunk.page_number,
            total_pages=chunk.total_pages,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            char_count=chunk.char_count,
            embedding=[float(val) for val in embedding],
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
