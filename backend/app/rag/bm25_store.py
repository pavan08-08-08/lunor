import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from rank_bm25 import BM25Okapi

from app.rag.chunker import ChunkDocument


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    """Encapsulates a BM25Okapi index and associated chunk metadata records."""
    bm25: BM25Okapi
    chunks: list[dict]


def build_bm25_index(chunks: list[ChunkDocument]) -> BM25Index:
    """Build a BM25 lexical index from a list of ChunkDocument objects.

    Args:
        chunks: List of ChunkDocument objects.

    Returns:
        A BM25Index instance containing the fitted BM25 model and chunk metadata.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot build BM25 index from empty chunk list")

    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    chunk_records = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number,
            "total_pages": chunk.total_pages,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "char_count": chunk.char_count,
        }
        for chunk in chunks
    ]

    return BM25Index(bm25=bm25, chunks=chunk_records)


def save_bm25_index(index: BM25Index, path: str | Path) -> None:
    """Save the BM25 index to disk via pickle.

    Args:
        index: The BM25Index instance to serialize.
        path: File or directory path where the index should be stored.
              If a directory is given, saves as bm25.pkl inside it.
    """
    target = Path(path)
    if target.is_dir() or target.suffix == "":
        target = target / "bm25.pkl"

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path: str | Path) -> BM25Index:
    """Load a locally persisted BM25 index from disk.

    Args:
        path: File or directory path containing bm25.pkl.

    Returns:
        The deserialized BM25Index instance.

    Raises:
        FileNotFoundError: If the target file does not exist.
        TypeError: If the deserialized object is not a BM25Index.
    """
    target = Path(path)
    if target.is_dir() or target.suffix == "":
        target = target / "bm25.pkl"

    if not target.exists():
        raise FileNotFoundError(f"BM25 index not found at {target}")

    with open(target, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, BM25Index):
        raise TypeError(f"Expected BM25Index object, got {type(index)}")

    return index


def bm25_search(
    index: BM25Index,
    query: str,
    k: int = 5,
) -> list[tuple[dict, float]]:
    """Perform BM25 lexical search for a given query string.

    Args:
        index: BM25Index instance.
        query: Raw user query string.
        k: Maximum number of top results to return.

    Returns:
        List of (chunk_dict, score) tuples sorted by descending score.
        Chunks with score <= 0 are excluded.
    """
    if not query or not query.strip():
        return []

    if k <= 0:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = index.bm25.get_scores(tokens)

    # Pair chunks with scores
    scored = [
        (chunk, float(score))
        for chunk, score in zip(index.chunks, scores)
        if score > 0.0
    ]

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
