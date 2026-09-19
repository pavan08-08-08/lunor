from pathlib import Path
import sys
import unittest

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import CHUNK_OVERLAP, CHUNK_SIZE, ChunkDocument, chunk_pages
from app.rag.loader import PageDocument


def _make_page(
    page_number: int = 1,
    text: str = "Sample content",
    total_pages: int = 1,
    is_empty: bool = False,
    doc_id: str = "doc_123",
    source_filename: str = "test.pdf",
) -> PageDocument:
    """Helper to create a PageDocument for testing."""
    return PageDocument(
        source_filename=source_filename,
        page_number=page_number,
        total_pages=total_pages,
        text=text,
        char_count=len(text),
        is_empty=is_empty,
        doc_id=doc_id,
    )


class TestChunker(unittest.TestCase):
    def test_short_page_produces_one_chunk(self):
        """A page shorter than CHUNK_SIZE should produce exactly one chunk."""
        text = "This is a short paragraph that easily fits within a single chunk."
        page = _make_page(text=text)

        chunks = chunk_pages([page])

        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], ChunkDocument)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].char_count, len(text))

    def test_long_page_produces_multiple_chunks(self):
        """A page longer than CHUNK_SIZE should produce multiple chunks."""
        # Generate text with sentences exceeding CHUNK_SIZE (1000)
        sentence = "Sentence describing a concept in the document. "
        text = sentence * 40  # ~1880 chars
        page = _make_page(text=text)

        chunks = chunk_pages([page])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, CHUNK_SIZE)

    def test_chunk_overlap_is_present(self):
        """Consecutive chunks from a split page should share overlapping text."""
        sentence = "This is a detailed sentence designed to verify overlap between split chunks. "
        text = sentence * 30  # ~2300 chars
        page = _make_page(text=text)

        chunks = chunk_pages([page])

        self.assertGreater(len(chunks), 1)
        # Check consecutive pairs for overlap
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].text
            chunk2_text = chunks[i + 1].text

            # Look for suffix of chunk1 that appears at prefix of chunk2
            overlap_found = False
            # Check overlap substrings of reasonable length up to CHUNK_OVERLAP
            for check_len in range(min(50, len(chunk1_text)), 0, -5):
                tail = chunk1_text[-check_len:]
                if tail in chunk2_text:
                    overlap_found = True
                    break
            self.assertTrue(
                overlap_found,
                f"Expected overlap between chunk {i} and {i + 1}",
            )

    def test_all_chunks_preserve_correct_page_number(self):
        """All chunks produced from a page must retain that page's page_number."""
        text = "Word word word. " * 150  # Long text
        page = _make_page(page_number=5, total_pages=10, text=text)

        chunks = chunk_pages([page])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.page_number, 5)
            self.assertEqual(chunk.total_pages, 10)

    def test_metadata_is_preserved(self):
        """ChunkDocument should correctly inherit all metadata from PageDocument."""
        text = "Short text for metadata testing."
        page = _make_page(
            page_number=3,
            total_pages=7,
            doc_id="abc123stablehash",
            source_filename="research_report.pdf",
            text=text,
        )

        chunks = chunk_pages([page])

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.doc_id, "abc123stablehash")
        self.assertEqual(chunk.source_filename, "research_report.pdf")
        self.assertEqual(chunk.page_number, 3)
        self.assertEqual(chunk.total_pages, 7)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.char_count, len(text))
        self.assertEqual(chunk.text, text)

    def test_chunk_index_starts_at_zero_for_each_page(self):
        """chunk_index must start at 0 independently for each page."""
        long_text = "Paragraph with enough text to split into multiple chunks. " * 30
        page1 = _make_page(page_number=1, text=long_text)
        page2 = _make_page(page_number=2, text=long_text)

        chunks = chunk_pages([page1, page2])

        p1_chunks = [c for c in chunks if c.page_number == 1]
        p2_chunks = [c for c in chunks if c.page_number == 2]

        self.assertGreater(len(p1_chunks), 1)
        self.assertGreater(len(p2_chunks), 1)

        self.assertEqual([c.chunk_index for c in p1_chunks], list(range(len(p1_chunks))))
        self.assertEqual([c.chunk_index for c in p2_chunks], list(range(len(p2_chunks))))

    def test_chunk_ids_are_unique(self):
        """All chunk IDs across all pages must be unique."""
        long_text = "Content for generating multiple distinct chunks. " * 30
        page1 = _make_page(page_number=1, text=long_text)
        page2 = _make_page(page_number=2, text=long_text)

        chunks = chunk_pages([page1, page2])

        chunk_ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_empty_pages_are_skipped(self):
        """Pages flagged with is_empty=True should be skipped."""
        empty_page = _make_page(page_number=1, is_empty=True, text="")
        valid_page = _make_page(page_number=2, text="Valid content on page 2")

        chunks = chunk_pages([empty_page, valid_page])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 2)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_whitespace_only_pages_are_skipped(self):
        """Pages with only whitespace/newlines should be defensively skipped."""
        whitespace_page = _make_page(
            page_number=1,
            is_empty=False,  # Even if flag is False, text.strip() is empty
            text="   \n\t  \n  ",
        )
        valid_page = _make_page(page_number=2, text="Real content here")

        chunks = chunk_pages([whitespace_page, valid_page])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 2)

    def test_multiple_pages_chunked_independently(self):
        """Pages must be chunked independently without cross-page text concatenation."""
        page1 = _make_page(page_number=1, text="Page 1 independent unique sentence.")
        page2 = _make_page(page_number=2, text="Page 2 distinct unrelated sentence.")

        chunks = chunk_pages([page1, page2])

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertIn("Page 1", chunks[0].text)
        self.assertNotIn("Page 2", chunks[0].text)

        self.assertEqual(chunks[1].page_number, 2)
        self.assertIn("Page 2", chunks[1].text)
        self.assertNotIn("Page 1", chunks[1].text)

    def test_only_empty_pages_returns_empty_list(self):
        """A document where all pages are empty or whitespace returns an empty list."""
        pages = [
            _make_page(page_number=1, is_empty=True, text=""),
            _make_page(page_number=2, is_empty=False, text="    \n "),
            _make_page(page_number=3, is_empty=True, text=""),
        ]

        chunks = chunk_pages(pages)

        self.assertEqual(chunks, [])

    def test_unicode_text_does_not_break(self):
        """Unicode characters (emojis, accents, CJK, symbols) should be handled cleanly."""
        unicode_text = (
            "🚀 Lunor Knowledge Assistant supports multilingual text: "
            "日本語テキスト, español con tildes y ñ, and math symbols: ∑, ∏, ∫, 𝒪(n log n). "
        ) * 10
        page = _make_page(text=unicode_text)

        chunks = chunk_pages([page])

        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.char_count, len(chunk.text))
            self.assertIn("日本語テキスト", chunk.text)


if __name__ == "__main__":
    unittest.main()
