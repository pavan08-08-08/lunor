import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from app.rag.embeddings import EMBEDDING_DIMENSION, EmbeddedChunk, get_embedding_model
from app.rag.vector_store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
    similarity_search,
)


def _make_embedded_chunk(
    chunk_id: str = "doc1_p1_c0",
    doc_id: str = "doc1",
    source_filename: str = "sample.pdf",
    page_number: int = 1,
    total_pages: int = 1,
    chunk_index: int = 0,
    text: str = "Sample chunk text for vector store test.",
    embedding: list[float] | None = None,
) -> EmbeddedChunk:
    """Helper to create an EmbeddedChunk for testing."""
    if embedding is None:
        # Create a unit vector along axis 0
        embedding = [0.0] * EMBEDDING_DIMENSION
        embedding[0] = 1.0

    return EmbeddedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_number=page_number,
        total_pages=total_pages,
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
        embedding=embedding,
    )


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Create distinct unit vectors for testing
        def make_unit_vec(axis: int) -> list[float]:
            v = [0.0] * EMBEDDING_DIMENSION
            v[axis % EMBEDDING_DIMENSION] = 1.0
            return v

        self.chunks = [
            _make_embedded_chunk(
                chunk_id="chunk_0",
                doc_id="doc_A",
                source_filename="doc_A.pdf",
                page_number=1,
                total_pages=2,
                chunk_index=0,
                text="Information on machine learning techniques.",
                embedding=make_unit_vec(0),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_1",
                doc_id="doc_A",
                source_filename="doc_A.pdf",
                page_number=1,
                total_pages=2,
                chunk_index=1,
                text="Detailed review of deep neural network architectures.",
                embedding=make_unit_vec(1),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_2",
                doc_id="doc_A",
                source_filename="doc_A.pdf",
                page_number=2,
                total_pages=2,
                chunk_index=0,
                text="Evaluation metrics for generative language models.",
                embedding=make_unit_vec(2),
            ),
        ]

    def test_build_vector_store_success(self):
        """Building a store from several EmbeddedChunk objects succeeds."""
        store = build_vector_store(self.chunks)
        self.assertIsNotNone(store)
        self.assertIsNotNone(store.index)

    def test_stored_vectors_count_matches_input_chunks(self):
        """The number of indexed vectors must match the number of input chunks."""
        store = build_vector_store(self.chunks)
        self.assertEqual(store.index.ntotal, len(self.chunks))

    def test_empty_input_raises_value_error(self):
        """Passing an empty chunk list must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            build_vector_store([])
        self.assertIn("empty chunk list", str(ctx.exception).lower())

    def test_metadata_preservation(self):
        """All metadata fields must be preserved on the stored Documents."""
        store = build_vector_store(self.chunks)
        # Search using chunk_0's exact vector
        results = similarity_search(store, self.chunks[0].embedding, k=1)
        self.assertEqual(len(results), 1)

        doc, score = results[0]
        self.assertEqual(doc.page_content, self.chunks[0].text)
        expected_meta = {
            "chunk_id": "chunk_0",
            "doc_id": "doc_A",
            "source_filename": "doc_A.pdf",
            "page_number": 1,
            "total_pages": 2,
            "chunk_index": 0,
            "char_count": len(self.chunks[0].text),
        }
        for k, v in expected_meta.items():
            self.assertIn(k, doc.metadata)
            self.assertEqual(doc.metadata[k], v)

    def test_similarity_search_returns_results(self):
        """similarity_search accepts an embedding vector and returns non-empty results."""
        store = build_vector_store(self.chunks)
        results = similarity_search(store, self.chunks[1].embedding, k=2)
        self.assertGreater(len(results), 0)

    def test_results_contain_document_and_score(self):
        """Each search result must be a (Document, float) tuple."""
        store = build_vector_store(self.chunks)
        results = similarity_search(store, self.chunks[0].embedding, k=2)

        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            doc, score = item
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(score, (float, int))

    def test_k_limits_result_count(self):
        """k=2 returns at most 2 results even when more vectors exist."""
        store = build_vector_store(self.chunks)
        results = similarity_search(store, self.chunks[0].embedding, k=2)
        self.assertEqual(len(results), 2)

    def test_k_greater_than_stored_chunks(self):
        """k greater than the number of stored chunks does not crash and returns all available chunks."""
        store = build_vector_store(self.chunks)  # 3 chunks
        results = similarity_search(store, self.chunks[0].embedding, k=10)
        self.assertEqual(len(results), 3)

    def test_save_and_load_round_trip(self):
        """Save -> load round trip produces a functional store with identical vector count."""
        store = build_vector_store(self.chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_vector_store(store, tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "index.faiss")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "index.pkl")))

            loaded_store = load_vector_store(tmp_dir)
            self.assertEqual(loaded_store.index.ntotal, store.index.ntotal)

            # Check similarity search works on loaded store
            results = similarity_search(loaded_store, self.chunks[0].embedding, k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0].page_content, self.chunks[0].text)

    def test_loaded_store_preserves_metadata(self):
        """Loaded store retains all chunk metadata accurately."""
        store = build_vector_store(self.chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_vector_store(store, tmp_dir)
            loaded_store = load_vector_store(tmp_dir)

            results = similarity_search(loaded_store, self.chunks[2].embedding, k=1)
            doc, _ = results[0]
            self.assertEqual(doc.metadata["chunk_id"], "chunk_2")
            self.assertEqual(doc.metadata["page_number"], 2)
            self.assertEqual(doc.metadata["total_pages"], 2)
            self.assertEqual(doc.metadata["source_filename"], "doc_A.pdf")

    def test_semantic_retrieval(self):
        """Real semantic retrieval test using actual embedding model."""
        model = get_embedding_model()

        chunk_1_text = "Customers may request a refund within thirty days of purchase."
        chunk_2_text = "Standard shipping normally takes three to five business days."
        chunk_3_text = "Users can reset their password from the account settings page."

        chunks = [
            _make_embedded_chunk(
                chunk_id="chunk_refund",
                text=chunk_1_text,
                embedding=model.embed_query(chunk_1_text),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_shipping",
                text=chunk_2_text,
                embedding=model.embed_query(chunk_2_text),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_password",
                text=chunk_3_text,
                embedding=model.embed_query(chunk_3_text),
            ),
        ]

        store = build_vector_store(chunks)

        query = "How can I get my money back?"
        query_vector = model.embed_query(query)

        results = similarity_search(store, query_vector, k=3)

        self.assertGreater(len(results), 0)
        top_doc, top_score = results[0]

        # The refund chunk must be the most similar to getting money back
        self.assertEqual(top_doc.metadata["chunk_id"], "chunk_refund")
        self.assertEqual(top_doc.page_content, chunk_1_text)
        self.assertGreater(top_score, 0.0)


if __name__ == "__main__":
    unittest.main()
