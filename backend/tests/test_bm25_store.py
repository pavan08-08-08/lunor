import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import ChunkDocument
from app.rag.bm25_store import (
    BM25Index,
    build_bm25_index,
    save_bm25_index,
    load_bm25_index,
    bm25_search,
    tokenize,
)


class TestBM25Store(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            ChunkDocument(
                chunk_id="doc1_p1_c0",
                doc_id="doc1",
                source_filename="paper1.pdf",
                page_number=1,
                total_pages=3,
                chunk_index=0,
                text="An Explainable Machine Learning Framework for Urban Road Safety Risk Prediction.",
                char_count=82,
            ),
            ChunkDocument(
                chunk_id="doc1_p1_c1",
                doc_id="doc1",
                source_filename="paper1.pdf",
                page_number=1,
                total_pages=3,
                chunk_index=1,
                text="Section B. Research Gap: While severity models exist, spatial transfer remains unaddressed.",
                char_count=91,
            ),
            ChunkDocument(
                chunk_id="doc1_p2_c0",
                doc_id="doc1",
                source_filename="paper1.pdf",
                page_number=2,
                total_pages=3,
                chunk_index=0,
                text="Data preprocessing and feature engineering for traffic flow and spatiotemporal features.",
                char_count=89,
            ),
        ]

    def test_bm25_index_build(self):
        """BM25 index builds successfully from ChunkDocument list with matching record counts."""
        index = build_bm25_index(self.chunks)
        self.assertIsInstance(index, BM25Index)
        self.assertEqual(len(index.chunks), 3)
        self.assertEqual(index.chunks[0]["chunk_id"], "doc1_p1_c0")
        self.assertEqual(index.chunks[1]["chunk_id"], "doc1_p1_c1")
        self.assertEqual(index.chunks[2]["chunk_id"], "doc1_p2_c0")

    def test_bm25_exact_phrase_retrieval(self):
        """Query 'research gap' ranks the chunk containing 'B. Research Gap' as the top result."""
        index = build_bm25_index(self.chunks)
        results = bm25_search(index, query="research gap", k=3)

        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk["chunk_id"], "doc1_p1_c1")
        self.assertIn("Research Gap", top_chunk["text"])
        self.assertGreater(score, 0.0)

    def test_bm25_save_load(self):
        """Saved BM25 index reloads cleanly and yields identical search scores."""
        index = build_bm25_index(self.chunks)
        original_results = bm25_search(index, query="traffic flow", k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_bm25_index(index, tmpdir)
            loaded_index = load_bm25_index(tmpdir)

            loaded_results = bm25_search(loaded_index, query="traffic flow", k=3)
            self.assertEqual(len(loaded_results), len(original_results))
            self.assertEqual(loaded_results[0][0]["chunk_id"], original_results[0][0]["chunk_id"])
            self.assertAlmostEqual(loaded_results[0][1], original_results[0][1], places=5)

    def test_bm25_empty_chunks_raises_value_error(self):
        """Building index from empty chunk list must raise ValueError."""
        with self.assertRaises(ValueError):
            build_bm25_index([])

    def test_bm25_empty_query(self):
        """Empty or whitespace-only queries return an empty result list."""
        index = build_bm25_index(self.chunks)
        self.assertEqual(bm25_search(index, ""), [])
        self.assertEqual(bm25_search(index, "   \t  "), [])


if __name__ == "__main__":
    unittest.main()
