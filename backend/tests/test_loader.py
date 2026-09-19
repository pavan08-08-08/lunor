import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from app.rag.loader import PageDocument, PDFLoadError, load_pdf


def _create_test_pdf(pages_text: list[str], output_path: str, password: str | None = None) -> str:
    """Helper to generate a temporary PDF with specified page contents."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)

    if password:
        doc.save(output_path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        doc.save(output_path)
    doc.close()
    return output_path


class TestPDFLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normal_multi_page_pdf(self):
        """Test standard extraction from a multi-page PDF."""
        pdf_path = os.path.join(self.temp_dir.name, "multi_page.pdf")
        page_texts = ["First page text", "Second page content", "Third page final notes"]
        _create_test_pdf(page_texts, pdf_path)

        docs = load_pdf(pdf_path)

        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, PageDocument)
            self.assertIn(page_texts[i], doc.text)
            self.assertEqual(doc.char_count, len(doc.text))
            self.assertFalse(doc.is_empty)
            self.assertEqual(doc.total_pages, 3)

    def test_page_numbering(self):
        """Test that page numbering is 1-indexed and matches document order."""
        pdf_path = os.path.join(self.temp_dir.name, "numbered.pdf")
        _create_test_pdf(["Page 1", "Page 2", "Page 3", "Page 4"], pdf_path)

        docs = load_pdf(pdf_path)

        self.assertEqual(len(docs), 4)
        page_numbers = [doc.page_number for doc in docs]
        self.assertEqual(page_numbers, [1, 2, 3, 4])
        for doc in docs:
            self.assertEqual(doc.total_pages, 4)

    def test_metadata(self):
        """Test document metadata including source_filename and stable doc_id."""
        filename = "my_custom_document.pdf"
        pdf_path = os.path.join(self.temp_dir.name, filename)
        _create_test_pdf(["Doc A content page 1", "Doc A content page 2"], pdf_path)

        docs = load_pdf(pdf_path)

        for doc in docs:
            # Filename should only be the basename, not the full directory path
            self.assertEqual(doc.source_filename, filename)
            self.assertNotIn(self.temp_dir.name, doc.source_filename)
            # doc_id should be attached to every PageDocument and be identical
            self.assertEqual(doc.doc_id, docs[0].doc_id)
            self.assertEqual(len(doc.doc_id), 64)

        # Stable doc_id: reloading the same file returns the same doc_id
        docs_reloaded = load_pdf(pdf_path)
        self.assertEqual(docs_reloaded[0].doc_id, docs[0].doc_id)

        # Different document content yields a different doc_id
        different_pdf_path = os.path.join(self.temp_dir.name, "other_doc.pdf")
        _create_test_pdf(["Completely different content"], different_pdf_path)
        other_docs = load_pdf(different_pdf_path)
        self.assertNotEqual(other_docs[0].doc_id, docs[0].doc_id)

    def test_empty_page_handling(self):
        """Test that empty pages are preserved and flagged with is_empty=True."""
        pdf_path = os.path.join(self.temp_dir.name, "empty_pages.pdf")
        # Page 1: normal text, Page 2: completely empty, Page 3: whitespace only
        _create_test_pdf(["Valid text on page 1", "", "   \n\t  \n"], pdf_path)

        docs = load_pdf(pdf_path)

        self.assertEqual(len(docs), 3)

        # Page 1: non-empty
        self.assertEqual(docs[0].page_number, 1)
        self.assertFalse(docs[0].is_empty)
        self.assertIn("Valid text", docs[0].text)

        # Page 2: completely blank
        self.assertEqual(docs[1].page_number, 2)
        self.assertTrue(docs[1].is_empty)
        self.assertEqual(docs[1].text.strip(), "")

        # Page 3: whitespace only
        self.assertEqual(docs[2].page_number, 3)
        self.assertTrue(docs[2].is_empty)
        self.assertEqual(docs[2].text.strip(), "")

    def test_missing_file(self):
        """Test that missing file raises PDFLoadError."""
        non_existent_path = os.path.join(self.temp_dir.name, "does_not_exist.pdf")
        with self.assertRaises(PDFLoadError) as ctx:
            load_pdf(non_existent_path)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_invalid_pdf(self):
        """Test that invalid/corrupted file raises PDFLoadError."""
        corrupt_path = os.path.join(self.temp_dir.name, "corrupt.pdf")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid pdf header or content")

        with self.assertRaises(PDFLoadError):
            load_pdf(corrupt_path)

    def test_encrypted_pdf(self):
        """Test that password-protected/encrypted PDF raises PDFLoadError."""
        encrypted_path = os.path.join(self.temp_dir.name, "protected.pdf")
        _create_test_pdf(["Confidential text"], encrypted_path, password="secret_password")

        with self.assertRaises(PDFLoadError) as ctx:
            load_pdf(encrypted_path)
        self.assertIn("encrypted", str(ctx.exception).lower())

    def test_page_level_extraction_failure_resilience(self):
        """Test that failure on a single page does not terminate document ingestion."""
        pdf_path = os.path.join(self.temp_dir.name, "partial_fail.pdf")
        _create_test_pdf(["Page 1 good", "Page 2 will fail", "Page 3 good"], pdf_path)

        original_load_page = pymupdf.Document.load_page

        def mock_load_page(doc_self, page_num):
            if page_num == 1:
                raise RuntimeError("Simulated corruption on page 2")
            return original_load_page(doc_self, page_num)

        with patch.object(pymupdf.Document, "load_page", side_effect=mock_load_page, autospec=True):
            docs = load_pdf(pdf_path)

        self.assertEqual(len(docs), 3)
        # Page 1 succeeded
        self.assertIn("Page 1 good", docs[0].text)
        self.assertFalse(docs[0].is_empty)

        # Page 2 failed but is preserved with empty text
        self.assertEqual(docs[1].page_number, 2)
        self.assertEqual(docs[1].text, "")
        self.assertEqual(docs[1].char_count, 0)
        self.assertTrue(docs[1].is_empty)

        # Page 3 succeeded
        self.assertIn("Page 3 good", docs[2].text)
        self.assertFalse(docs[2].is_empty)


if __name__ == "__main__":
    unittest.main()
