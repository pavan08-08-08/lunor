from dataclasses import dataclass
from langchain_community.vectorstores import FAISS

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import similarity_search
from app.rag.bm25_store import BM25Index, bm25_search, tokenize

# Initial heuristic relevance threshold.
# This threshold should be tuned later using representative evaluation questions
# against the actual document corpus, as 0.35 is an initial heuristic and not universally optimal.
DEFAULT_RELEVANCE_THRESHOLD: float = 0.35

# Common stop words used to extract meaningful content terms from user queries for lexical qualification
STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
}


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
    bm25_index: BM25Index | None = None,
    k: int = 5,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks using hybrid (semantic + BM25) or semantic-only retrieval.

    Args:
        query: Raw user query string.
        store: FAISS vector store containing indexed documents.
        bm25_index: Optional BM25Index for lexical retrieval. If None, falls back to semantic-only.
        k: Maximum number of candidates to retrieve. Must be positive.
        threshold: Minimum cosine similarity threshold for semantic evidence.

    Returns:
        List of RetrievedChunk objects meeting relevance criteria, ordered by descending score.

    Raises:
        ValueError: If query is empty/whitespace-only or if k <= 0.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # If no BM25 index is provided, execute existing semantic-only retrieval
    if bm25_index is None:
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

    # Hybrid Retrieval (FAISS + BM25 + Reciprocal Rank Fusion)
    semantic_k = max(k, 20)
    lexical_k = max(k, 20)

    model = get_embedding_model()
    query_vector = model.embed_query(query)
    semantic_results = similarity_search(store=store, query_embedding=query_vector, k=semantic_k)
    lexical_results = bm25_search(index=bm25_index, query=query, k=lexical_k)

    rrf_k = 60
    rrf_scores: dict[str, float] = {}
    chunk_info: dict[str, dict] = {}
    semantic_scores: dict[str, float] = {}
    lexical_ranks: dict[str, int] = {}

    for rank, (doc, score) in enumerate(semantic_results, 1):
        cid = doc.metadata["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        semantic_scores[cid] = score
        chunk_info[cid] = {
            "chunk_id": cid,
            "doc_id": doc.metadata["doc_id"],
            "source_filename": doc.metadata["source_filename"],
            "page_number": doc.metadata["page_number"],
            "total_pages": doc.metadata["total_pages"],
            "chunk_index": doc.metadata["chunk_index"],
            "text": doc.page_content,
        }

    for rank, (rec, score) in enumerate(lexical_results, 1):
        cid = rec["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        lexical_ranks[cid] = rank
        if cid not in chunk_info:
            chunk_info[cid] = {
                "chunk_id": cid,
                "doc_id": rec["doc_id"],
                "source_filename": rec["source_filename"],
                "page_number": rec["page_number"],
                "total_pages": rec["total_pages"],
                "chunk_index": rec["chunk_index"],
                "text": rec["text"],
            }

    # Extract meaningful content tokens for lexical qualification
    q_tokens = set(tokenize(query))
    content_tokens = q_tokens - STOP_WORDS
    if not content_tokens:
        content_tokens = q_tokens

    # Relevance qualification gate
    qualified_chunks: list[RetrievedChunk] = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True):
        sem_score = semantic_scores.get(cid, 0.0)
        lex_rank = lexical_ranks.get(cid, 999)
        meta = chunk_info[cid]
        c_tokens = set(tokenize(meta["text"]))
        has_overlap = bool(content_tokens & c_tokens)

        # Dual evidence: semantic confidence OR top-rank lexical match with content overlap
        passes_sem = sem_score >= threshold
        passes_lex = lex_rank <= 3 and has_overlap

        if passes_sem or passes_lex:
            qualified_chunks.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    source_filename=meta["source_filename"],
                    page_number=meta["page_number"],
                    total_pages=meta["total_pages"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    similarity_score=rrf_score,
                )
            )

    return qualified_chunks[:k]
