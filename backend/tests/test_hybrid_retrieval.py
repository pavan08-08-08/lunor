import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import ChunkDocument, chunk_pages
from app.rag.embeddings import embed_chunks
from app.rag.loader import load_pdf
from app.rag.vector_store import build_vector_store
from app.rag.bm25_store import build_bm25_index
from app.rag.retriever import retrieve, DEFAULT_RELEVANCE_THRESHOLD, RetrievedChunk


class TestHybridRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate sample PDF
        sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "uploads" / "sample1_paper copy.pdf"
        if not sample_path.exists():
            sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "documents" / "sample1_paper copy.pdf"
        cls.sample_path = sample_path

        if sample_path.exists():
            pages = load_pdf(str(sample_path))
            cls.sample_chunks = chunk_pages(pages)
            cls.sample_embedded = embed_chunks(cls.sample_chunks)
            cls.sample_faiss = build_vector_store(cls.sample_embedded)
            cls.sample_bm25 = build_bm25_index(cls.sample_chunks)
        else:
            cls.sample_chunks = []
            cls.sample_embedded = []
            cls.sample_faiss = None
            cls.sample_bm25 = None

    def test_semantic_threshold_unchanged(self):
        """DEFAULT_RELEVANCE_THRESHOLD must remain strictly 0.35."""
        self.assertEqual(DEFAULT_RELEVANCE_THRESHOLD, 0.35)

    def test_empty_query_raises_value_error(self):
        """Empty or whitespace query must raise ValueError."""
        dummy_store = MagicMock()
        with self.assertRaises(ValueError):
            retrieve("", dummy_store)
        with self.assertRaises(ValueError):
            retrieve("   \n\t  ", dummy_store)

    def test_k_validation(self):
        """k <= 0 must raise ValueError."""
        dummy_store = MagicMock()
        with self.assertRaises(ValueError):
            retrieve("test query", dummy_store, k=0)
        with self.assertRaises(ValueError):
            retrieve("test query", dummy_store, k=-1)

    def test_retrieval_without_bm25_fallback(self):
        """When bm25_index is None, retriever performs backward-compatible semantic-only search."""
        if self.sample_faiss is None:
            self.skipTest("Sample PDF not found")

        # Query that satisfies semantic threshold
        results = retrieve(
            query="What machine learning framework is proposed for traffic risk prediction?",
            store=self.sample_faiss,
            bm25_index=None,
            k=5,
            threshold=0.35,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, RetrievedChunk)
            self.assertGreaterEqual(r.similarity_score, 0.35)

    def test_rrf_fusion(self):
        """Reciprocal Rank Fusion accurately fuses controlled semantic and lexical rankings."""
        # Chunk A is rank 1 in semantic (1/(60+1) = 1/61 = 0.01639), rank 2 in lexical (1/(60+2) = 1/62 = 0.01613)
        # Total RRF for A = 0.03252
        # Chunk B is rank 2 in semantic (1/62 = 0.01613), not in lexical -> Total RRF = 0.01613
        # Chunk C is not in semantic, rank 1 in lexical (1/61 = 0.01639) -> Total RRF = 0.01639
        # Expected order: A (0.03252), C (0.01639), B (0.01613)

        chunk_a = ChunkDocument(
            chunk_id="cA", doc_id="d1", source_filename="f.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Machine learning traffic model research gap analysis", char_count=50
        )
        chunk_b = ChunkDocument(
            chunk_id="cB", doc_id="d1", source_filename="f.pdf", page_number=1, total_pages=1, chunk_index=1,
            text="Neural networks deep learning traffic framework", char_count=45
        )
        chunk_c = ChunkDocument(
            chunk_id="cC", doc_id="d1", source_filename="f.pdf", page_number=1, total_pages=1, chunk_index=2,
            text="Research gap identified in spatial transferability", char_count=48
        )

        all_chunks = [chunk_a, chunk_b, chunk_c]
        embedded = embed_chunks(all_chunks)
        store = build_vector_store(embedded)
        bm25_idx = build_bm25_index(all_chunks)

        results = retrieve(
            query="traffic research gap",
            store=store,
            bm25_index=bm25_idx,
            k=3,
        )

        self.assertGreater(len(results), 0)
        # chunk_a contains both semantic terms and lexical terms, must rank first
        self.assertEqual(results[0].chunk_id, "cA")

    def test_hybrid_retrieval_preserves_semantic_query(self):
        """Hybrid retrieval continues returning high-scoring chunks for known semantic queries."""
        if self.sample_faiss is None or self.sample_bm25 is None:
            self.skipTest("Sample PDF not found")

        query = "What machine learning framework is proposed for traffic risk prediction?"
        results = retrieve(
            query=query,
            store=self.sample_faiss,
            bm25_index=self.sample_bm25,
            k=5,
        )

        self.assertEqual(len(results), 5)
        # Top chunk should cite page 1
        self.assertEqual(results[0].page_number, 1)
        self.assertIn("Machine Learning Framework", results[0].text)

    def test_hybrid_retrieval_recovers_research_gap(self):
        """Regression test: 'What is the research gap?' successfully retrieves Section B. Research Gap chunk."""
        if self.sample_faiss is None or self.sample_bm25 is None:
            self.skipTest("Sample PDF not found")

        query = "What is the research gap?"

        # Verify old semantic-only retrieval failed:
        old_results = retrieve(
            query=query,
            store=self.sample_faiss,
            bm25_index=None,
            k=5,
            threshold=DEFAULT_RELEVANCE_THRESHOLD,
        )
        self.assertEqual(len(old_results), 0, "Semantic-only retrieval must fail due to 0.35 threshold")

        # Verify new hybrid retrieval succeeds:
        hybrid_results = retrieve(
            query=query,
            store=self.sample_faiss,
            bm25_index=self.sample_bm25,
            k=5,
        )

        self.assertGreater(len(hybrid_results), 0, "Hybrid retrieval must recover research gap chunk")
        gap_chunks = [r for r in hybrid_results if "Research Gap" in r.text or "research gap" in r.text.lower()]
        self.assertGreater(len(gap_chunks), 0, "At least one retrieved chunk must contain 'Research Gap'")
        # Verify it references page 1 where Section B. Research Gap is located
        self.assertTrue(any(r.page_number == 1 for r in gap_chunks))


if __name__ == "__main__":
    unittest.main()
