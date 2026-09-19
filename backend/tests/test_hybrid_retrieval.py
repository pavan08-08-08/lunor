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

    def test_bm25_relative_threshold_accepts_strong_and_rejects_distractor(self):
        """Chunks with BM25 >= 50% of top BM25 pass, while distractors below 50% are rejected."""
        # Chunk 1: target matching specific technical terms (high BM25)
        # Chunk 2: distractor matching only one incidental common term (low BM25 < 50%)
        # Chunk 3: background chunk ensuring corpus N >= 3 for positive BM25Okapi IDF
        chunk_target = ChunkDocument(
            chunk_id="c_target", doc_id="d1", source_filename="target.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Project Atlas evaluation achieved retrieval recall benchmark 0.82 metric", char_count=73
        )
        chunk_distractor = ChunkDocument(
            chunk_id="c_distractor", doc_id="d2", source_filename="distractor.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Project Borealis telemetry equipment monitoring sensor pipeline", char_count=65
        )
        chunk_bg = ChunkDocument(
            chunk_id="c_bg", doc_id="d3", source_filename="bg.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="General system documentation describing software environment", char_count=61
        )

        all_chunks = [chunk_target, chunk_distractor, chunk_bg]
        embedded = embed_chunks(all_chunks)
        store = build_vector_store(embedded)
        bm25_idx = build_bm25_index(all_chunks)

        # Query where both match 'project', but target matches 'recall', 'atlas', 'project'
        query = "What was the recall achieved by Project Atlas?"
        results = retrieve(
            query=query,
            store=store,
            bm25_index=bm25_idx,
            k=5,
            threshold=0.99,  # Force lexical qualification only by setting high semantic threshold
        )

        cids = [r.chunk_id for r in results]
        self.assertIn("c_target", cids, "Target chunk with strong BM25 score must qualify")
        self.assertNotIn("c_distractor", cids, "Distractor with <50% top BM25 score must be rejected")

    def test_exact_technical_term_passes_with_single_overlapping_token(self):
        """Exact technical acronym match passes lexical gate even with only one content token match."""
        chunk_rrf = ChunkDocument(
            chunk_id="c_rrf", doc_id="d1", source_filename="atlas.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Atlas uses Reciprocal Rank Fusion RRF ranking algorithm", char_count=56
        )
        chunk_other = ChunkDocument(
            chunk_id="c_other", doc_id="d2", source_filename="paper.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Urban transport safety predictions with gradient boosted trees", char_count=64
        )
        chunk_bg = ChunkDocument(
            chunk_id="c_bg", doc_id="d3", source_filename="bg.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="General statistical data analysis and exploratory research methods", char_count=68
        )

        all_chunks = [chunk_rrf, chunk_other, chunk_bg]
        embedded = embed_chunks(all_chunks)
        store = build_vector_store(embedded)
        bm25_idx = build_bm25_index(all_chunks)

        query = "What does RRF stand for?"
        results = retrieve(
            query=query,
            store=store,
            bm25_index=bm25_idx,
            k=5,
            threshold=0.99,  # Force lexical qualification only
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c_rrf")
        self.assertIn("RRF", results[0].text)

    def test_relative_semantic_margin_accepts_strong_and_rejects_distractor(self):
        """Candidates with semantic score >= 0.35 but < 70% of top semantic score are rejected."""
        # Query embedded with mock so we have exact deterministic semantic scores:
        # Candidate 1 (strong target): cosine similarity 0.70 (top_sem = 0.70)
        # Candidate 2 (distractor): cosine similarity 0.42 (0.42 >= 0.35, but 0.42 / 0.70 = 0.60 < 0.70)
        # Candidate 3 (background): cosine similarity 0.20
        chunk_target = ChunkDocument(
            chunk_id="c_top", doc_id="d1", source_filename="target.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="Project Atlas retrieval recall metric 0.82 benchmark", char_count=52
        )
        chunk_distractor = ChunkDocument(
            chunk_id="c_dist", doc_id="d2", source_filename="distractor.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="General overview of document processing architectures", char_count=54
        )
        chunk_bg = ChunkDocument(
            chunk_id="c_bg", doc_id="d3", source_filename="bg.pdf", page_number=1, total_pages=1, chunk_index=0,
            text="System telemetry and operating parameters description", char_count=54
        )

        all_chunks = [chunk_target, chunk_distractor, chunk_bg]
        embedded = embed_chunks(all_chunks)
        store = build_vector_store(embedded)
        bm25_idx = build_bm25_index(all_chunks)

        # Mock similarity_search to return controlled semantic scores:
        # top = 0.70, distractor = 0.42 (60% of top, above 0.35), bg = 0.20
        doc_top = MagicMock()
        doc_top.metadata = chunk_target.__dict__
        doc_top.page_content = chunk_target.text

        doc_dist = MagicMock()
        doc_dist.metadata = chunk_distractor.__dict__
        doc_dist.page_content = chunk_distractor.text

        doc_bg = MagicMock()
        doc_bg.metadata = chunk_bg.__dict__
        doc_bg.page_content = chunk_bg.text

        mock_sem_results = [(doc_top, 0.70), (doc_dist, 0.42), (doc_bg, 0.20)]

        with patch("app.rag.retriever.similarity_search", return_value=mock_sem_results):
            # Query has no lexical overlap with any chunk to test pure semantic gating
            query = "unmatched question with unique keywords"
            results = retrieve(
                query=query,
                store=store,
                bm25_index=bm25_idx,
                k=5,
            )

        cids = [r.chunk_id for r in results]
        self.assertIn("c_top", cids, "Strong top semantic candidate must qualify")
        self.assertNotIn(
            "c_dist",
            cids,
            "Distractor below 70% of top semantic score must be rejected despite exceeding 0.35",
        )


if __name__ == "__main__":
    unittest.main()
