from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from app.rag.embeddings import EmbeddedChunk, get_embedding_model
from app.rag.retriever import DEFAULT_RELEVANCE_THRESHOLD, RetrievedChunk, retrieve
from app.rag.vector_store import build_vector_store


def _make_embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    doc_id: str = "doc_test_1",
    source_filename: str = "sample_test.pdf",
    page_number: int = 1,
    total_pages: int = 3,
    chunk_index: int = 0,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk for testing."""
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


class TestRetriever(unittest.TestCase):
    def test_empty_query_raises_value_error(self):
        """Empty string query must raise ValueError before invoking embedding model."""
        mock_store = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            retrieve("", mock_store)
        self.assertIn("query cannot be empty", str(ctx.exception).lower())

    def test_whitespace_query_raises_value_error(self):
        """Whitespace-only query must raise ValueError before invoking embedding model."""
        mock_store = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            retrieve("   \t\n  ", mock_store)
        self.assertIn("query cannot be empty", str(ctx.exception).lower())

    def test_k_zero_raises_value_error(self):
        """k=0 must raise ValueError."""
        mock_store = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            retrieve("valid query", mock_store, k=0)
        self.assertIn("k must be a positive integer", str(ctx.exception).lower())

    def test_negative_k_raises_value_error(self):
        """Negative k must raise ValueError."""
        mock_store = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            retrieve("valid query", mock_store, k=-5)
        self.assertIn("k must be a positive integer", str(ctx.exception).lower())

    @patch("app.rag.retriever.similarity_search")
    def test_successful_retrieval_returns_retrieved_chunks(self, mock_search):
        """Successful retrieval returns a list of RetrievedChunk objects."""
        doc = Document(
            page_content="Sample document text.",
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "source_filename": "f.pdf",
                "page_number": 1,
                "total_pages": 1,
                "chunk_index": 0,
                "char_count": 21,
            },
        )
        mock_search.return_value = [(doc, 0.85)]
        mock_store = MagicMock()

        results = retrieve("sample question", mock_store, k=1, threshold=0.5)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].text, "Sample document text.")
        self.assertEqual(results[0].similarity_score, 0.85)

    @patch("app.rag.retriever.similarity_search")
    @patch("app.rag.retriever.get_embedding_model")
    def test_query_is_embedded_using_existing_model(self, mock_get_model, mock_search):
        """retrieve() embeds the raw query using the existing model singleton."""
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1] * 384
        mock_get_model.return_value = mock_model
        mock_search.return_value = []
        mock_store = MagicMock()

        retrieve("test search term", mock_store)
        mock_get_model.assert_called_once()
        mock_model.embed_query.assert_called_once_with("test search term")

    @patch("app.rag.retriever.similarity_search")
    def test_metadata_is_preserved(self, mock_search):
        """All required metadata fields must be preserved on RetrievedChunk."""
        doc = Document(
            page_content="Content about algorithms.",
            metadata={
                "chunk_id": "chunk_99",
                "doc_id": "doc_alpha",
                "source_filename": "algorithms.pdf",
                "page_number": 4,
                "total_pages": 10,
                "chunk_index": 2,
                "char_count": 25,
            },
        )
        mock_search.return_value = [(doc, 0.72)]
        mock_store = MagicMock()

        results = retrieve("algorithms query", mock_store, k=1, threshold=0.3)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.chunk_id, "chunk_99")
        self.assertEqual(res.doc_id, "doc_alpha")
        self.assertEqual(res.source_filename, "algorithms.pdf")
        self.assertEqual(res.page_number, 4)
        self.assertEqual(res.total_pages, 10)
        self.assertEqual(res.chunk_index, 2)
        self.assertEqual(res.text, "Content about algorithms.")

    @patch("app.rag.retriever.similarity_search")
    def test_similarity_score_is_preserved(self, mock_search):
        """The exact similarity score returned from vector store search is preserved."""
        doc = Document(
            page_content="Some content",
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "source_filename": "s.pdf",
                "page_number": 1,
                "total_pages": 1,
                "chunk_index": 0,
            },
        )
        mock_search.return_value = [(doc, 0.612345)]
        mock_store = MagicMock()

        results = retrieve("query", mock_store, k=1, threshold=0.1)
        self.assertAlmostEqual(results[0].similarity_score, 0.612345, places=5)

    @patch("app.rag.retriever.similarity_search")
    def test_results_ordered_by_descending_similarity(self, mock_search):
        """Results must be ordered strictly from highest similarity to lowest similarity."""
        docs = [
            (
                Document(
                    page_content="Text A",
                    metadata={"chunk_id": "cA", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 0},
                ),
                0.90,
            ),
            (
                Document(
                    page_content="Text B",
                    metadata={"chunk_id": "cB", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 1},
                ),
                0.75,
            ),
            (
                Document(
                    page_content="Text C",
                    metadata={"chunk_id": "cC", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 2},
                ),
                0.55,
            ),
        ]
        # Even if search returned in arbitrary order, retrieve sorts descending
        mock_search.return_value = [docs[1], docs[0], docs[2]]
        mock_store = MagicMock()

        results = retrieve("ordering query", mock_store, k=3, threshold=0.4)

        scores = [r.similarity_score for r in results]
        self.assertEqual(scores, [0.90, 0.75, 0.55])
        self.assertEqual([r.chunk_id for r in results], ["cA", "cB", "cC"])

    @patch("app.rag.retriever.similarity_search")
    def test_results_above_threshold_included(self, mock_search):
        """Results with similarity score >= threshold must be included."""
        doc1 = (Document(page_content="T1", metadata={"chunk_id": "c1", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 0}), 0.50)
        doc2 = (Document(page_content="T2", metadata={"chunk_id": "c2", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 1}), 0.35)
        mock_search.return_value = [doc1, doc2]
        mock_store = MagicMock()

        results = retrieve("threshold test", mock_store, k=2, threshold=0.35)

        self.assertEqual(len(results), 2)
        self.assertEqual([r.chunk_id for r in results], ["c1", "c2"])

    @patch("app.rag.retriever.similarity_search")
    def test_results_below_threshold_excluded(self, mock_search):
        """Results with similarity score < threshold must be excluded."""
        doc1 = (Document(page_content="T1", metadata={"chunk_id": "c1", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 0}), 0.60)
        doc2 = (Document(page_content="T2", metadata={"chunk_id": "c2", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 1}), 0.34)
        mock_search.return_value = [doc1, doc2]
        mock_store = MagicMock()

        results = retrieve("threshold test", mock_store, k=2, threshold=0.35)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")

    @patch("app.rag.retriever.similarity_search")
    def test_no_results_above_threshold_returns_empty_list(self, mock_search):
        """If all candidates are below threshold, returns [] without error."""
        doc = (Document(page_content="T", metadata={"chunk_id": "c", "doc_id": "d", "source_filename": "f", "page_number": 1, "total_pages": 1, "chunk_index": 0}), 0.20)
        mock_search.return_value = [doc]
        mock_store = MagicMock()

        results = retrieve("query", mock_store, k=5, threshold=0.35)

        self.assertEqual(results, [])

    def test_semantic_retrieval_controlled_dataset(self):
        """Real end-to-end semantic retrieval test using actual embedding model and FAISS store."""
        model = get_embedding_model()

        chunk_a_text = "Customers may request a refund within thirty days of purchase."
        chunk_b_text = "Standard shipping normally takes three to five business days."
        chunk_c_text = "Users can reset their password from the account settings page."

        chunks = [
            _make_embedded_chunk(
                chunk_id="chunk_A",
                text=chunk_a_text,
                embedding=model.embed_query(chunk_a_text),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_B",
                text=chunk_b_text,
                embedding=model.embed_query(chunk_b_text),
            ),
            _make_embedded_chunk(
                chunk_id="chunk_C",
                text=chunk_c_text,
                embedding=model.embed_query(chunk_c_text),
            ),
        ]

        store = build_vector_store(chunks)

        query = "How can I get my money back?"
        results = retrieve(query, store, k=3, threshold=0.35)

        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result.chunk_id, "chunk_A")
        self.assertEqual(top_result.text, chunk_a_text)
        self.assertGreater(top_result.similarity_score, 0.35)


if __name__ == "__main__":
    unittest.main()
