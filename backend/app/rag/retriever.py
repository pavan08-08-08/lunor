from dataclasses import dataclass
from langchain_community.vectorstores import FAISS

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import similarity_search

# Initial heuristic relevance threshold.
# This threshold should be tuned later using representative evaluation questions
# against the actual document corpus, as 0.35 is an initial heuristic and not universally optimal.
DEFAULT_RELEVANCE_THRESHOLD: float = 0.35


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with relevance score and metadata."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    text: str
    similarity_score: float


def retrieve(
    query: str,
    store: FAISS,
    k: int = 5,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks from the FAISS vector store for a given question.

    Args:
        query: Raw user query string.
        store: FAISS vector store containing indexed documents.
        k: Maximum number of candidates to retrieve. Must be positive.
        threshold: Minimum cosine similarity / inner-product threshold.

    Returns:
        List of RetrievedChunk objects meeting the threshold, ordered by descending similarity score.

    Raises:
        ValueError: If query is empty/whitespace-only or if k <= 0.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    model = get_embedding_model()
    query_vector = model.embed_query(query)

    search_results = similarity_search(store=store, query_embedding=query_vector, k=k)

    retrieved: list[RetrievedChunk] = []
    for doc, score in search_results:
        if score >= threshold:
            metadata = doc.metadata
            retrieved.append(
                RetrievedChunk(
                    chunk_id=metadata["chunk_id"],
                    doc_id=metadata["doc_id"],
                    source_filename=metadata["source_filename"],
                    page_number=metadata["page_number"],
                    total_pages=metadata["total_pages"],
                    chunk_index=metadata["chunk_index"],
                    text=doc.page_content,
                    similarity_score=score,
                )
            )

    retrieved.sort(key=lambda chunk: chunk.similarity_score, reverse=True)
    return retrieved
