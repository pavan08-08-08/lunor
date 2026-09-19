import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import ChunkDocument
from app.rag.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EmbeddedChunk,
    embed_chunks,
    get_embedding_model,
)


def _make_chunk(
    chunk_id: str = "doc1_p1_c0",
    doc_id: str = "doc1",
    source_filename: str = "test.pdf",
    page_number: int = 1,
    total_pages: int = 1,
    chunk_index: int = 0,
    text: str = "Artificial intelligence and retrieval augmented generation are transforming knowledge discovery.",
) -> ChunkDocument:
    """Helper to construct a ChunkDocument for testing."""
    return ChunkDocument(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_number=page_number,
        total_pages=total_pages,
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
    )


class TestEmbeddings(unittest.TestCase):
    def test_model_singleton_reuse(self):
        """get_embedding_model() must return the same instance across multiple calls."""
        model_1 = get_embedding_model()
        model_2 = get_embedding_model()
        self.assertIs(model_1, model_2)

    def test_empty_input_returns_empty_list_without_embedding_call(self):
        """Passing an empty list of chunks should return [] without invoking get_embedding_model."""
        with patch("app.rag.embeddings.get_embedding_model") as mock_get_model:
            result = embed_chunks([])
            self.assertEqual(result, [])
            mock_get_model.assert_not_called()

    def test_small_list_returns_matching_count_of_embedded_chunks(self):
        """Embedding a small batch should return exactly one EmbeddedChunk per input ChunkDocument."""
        chunks = [
            _make_chunk(chunk_id="c1", text="First chunk on deep learning."),
            _make_chunk(chunk_id="c2", text="Second chunk discussing neural representations."),
            _make_chunk(chunk_id="c3", text="Third chunk explaining dense vector retrieval."),
        ]

        embedded = embed_chunks(chunks)

        self.assertEqual(len(embedded), 3)
        for item in embedded:
            self.assertIsInstance(item, EmbeddedChunk)

    def test_embedding_dimension(self):
        """Each returned embedding vector must have dimension equal to EMBEDDING_DIMENSION (384)."""
        chunks = [_make_chunk(text="Vector dimensionality verification.")]
        embedded = embed_chunks(chunks)

        self.assertEqual(len(embedded), 1)
        self.assertEqual(len(embedded[0].embedding), EMBEDDING_DIMENSION)
        self.assertEqual(EMBEDDING_DIMENSION, 384)

    def test_embeddings_are_unit_normalized(self):
        """Embeddings must be approximately unit-normalized (L2 norm ~ 1.0)."""
        chunks = [
            _make_chunk(text="Sentence testing normalization property."),
            _make_chunk(text="Another text segment to ensure consistent L2 unit length."),
        ]
        embedded = embed_chunks(chunks)

        for item in embedded:
            l2_norm = math.sqrt(sum(x * x for x in item.embedding))
            self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_metadata_preservation(self):
        """All original ChunkDocument metadata must be preserved exactly."""
        chunk = _make_chunk(
            chunk_id="custom_chunk_id_42",
            doc_id="doc_xyz_987",
            source_filename="research_paper.pdf",
            page_number=5,
            total_pages=12,
            chunk_index=3,
            text="Precise metadata verification text.",
        )

        embedded = embed_chunks([chunk])

        self.assertEqual(len(embedded), 1)
        res = embedded[0]
        self.assertEqual(res.chunk_id, "custom_chunk_id_42")
        self.assertEqual(res.doc_id, "doc_xyz_987")
        self.assertEqual(res.source_filename, "research_paper.pdf")
        self.assertEqual(res.page_number, 5)
        self.assertEqual(res.total_pages, 12)
        self.assertEqual(res.chunk_index, 3)
        self.assertEqual(res.text, "Precise metadata verification text.")
        self.assertEqual(res.char_count, len("Precise metadata verification text."))
        self.assertIsInstance(res.embedding, list)
        self.assertTrue(all(isinstance(v, float) for v in res.embedding))

    def test_same_text_produces_identical_vectors(self):
        """Embedding the exact same text in separate chunks produces identical vector representations."""
        text = "Deterministic vector representation for identical semantic inputs."
        chunk_a = _make_chunk(chunk_id="ca", text=text)
        chunk_b = _make_chunk(chunk_id="cb", text=text)

        embedded = embed_chunks([chunk_a, chunk_b])

        self.assertEqual(len(embedded), 2)
        for val_a, val_b in zip(embedded[0].embedding, embedded[1].embedding):
            self.assertAlmostEqual(val_a, val_b, places=5)


if __name__ == "__main__":
    unittest.main()
