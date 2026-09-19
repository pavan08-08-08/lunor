from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.rag.embeddings import EmbeddedChunk, get_embedding_model


def build_vector_store(chunks: list[EmbeddedChunk]) -> FAISS:
    """Build a FAISS vector store from already-embedded chunks using inner-product similarity.

    Args:
        chunks: Non-empty list of EmbeddedChunk objects.

    Returns:
        FAISS vector store initialized with the precomputed vectors and documents.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunk list")

    text_embeddings = [(chunk.text, chunk.embedding) for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number,
            "total_pages": chunk.total_pages,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
        }
        for chunk in chunks
    ]

    return FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=get_embedding_model(),
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def save_vector_store(store: FAISS, path: str) -> None:
    """Save the FAISS vector store to a local directory path.

    Args:
        store: The FAISS vector store to save.
        path: Directory path where index files will be stored.
    """
    store.save_local(path)


def load_vector_store(path: str) -> FAISS:
    """Load a FAISS vector store from a local directory path.

    Args:
        path: Directory path containing the saved index files.

    Returns:
        The loaded FAISS vector store instance.
    """
    # This is safe here because the index is generated and controlled by this application's
    # own local ingestion pipeline and is not loaded from an untrusted source.
    return FAISS.load_local(
        folder_path=path,
        embeddings=get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def similarity_search(
    store: FAISS,
    query_embedding: list[float],
    k: int = 5,
) -> list[tuple[Document, float]]:
    """Perform exact inner-product similarity search on the vector store using a precomputed query embedding.

    Args:
        store: FAISS vector store instance.
        query_embedding: Precomputed query embedding vector.
        k: Maximum number of nearest results to retrieve.

    Returns:
        List of (Document, similarity_score) tuples ordered by descending similarity.
    """
    results = store.similarity_search_with_score_by_vector(
        embedding=query_embedding,
        k=k,
    )
    return [(doc, float(score)) for doc, score in results]
