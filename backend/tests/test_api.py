import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.api.chat import ChatResponse
from app.main import app
from app.rag.generator import (
    GeneratedAnswer,
    GenerationQuotaExceededError,
    GenerationUnavailableError,
    SourceCitation,
)
from app.rag.loader import PageDocument


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.uploads_dir = self.temp_path / "uploads"
        self.vectorstore_dir = self.temp_path / "vectorstore"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_endpoint(self):
        """GET /api/health must return 200 with status ok."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_without_vectorstore(self):
        """POST /api/chat when vector store does not exist must return HTTP 400."""
        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.post("/api/chat", json={"query": "What is overfitting?"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("no documents are indexed", response.json()["detail"].lower())

    @patch("app.api.chat.generate_answer")
    @patch("app.api.chat.retrieve")
    @patch("app.api.chat.load_bm25_index")
    @patch("app.api.chat.load_vector_store")
    def test_chat_success(self, mock_load, mock_bm25, mock_retrieve, mock_generate):
        """Valid query returns HTTP 200 with answer, sources, and context flag."""
        # Create dummy index files so the endpoint sees an existing vector store
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()
        (self.vectorstore_dir / "bm25.pkl").touch()

        mock_load.return_value = MagicMock()
        mock_bm25.return_value = MagicMock()
        mock_retrieve.return_value = [MagicMock()]
        mock_generate.return_value = GeneratedAnswer(
            answer="Overfitting occurs when a model learns noise.",
            sources=[
                SourceCitation(
                    source_filename="ml.pdf",
                    page_number=3,
                    chunk_id="ml_p3_c0",
                    text="Overfitting occurs when a model learns noise.",
                )
            ],
            has_sufficient_context=True,
        )

        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.post("/api/chat", json={"query": "What is overfitting?"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["answer"], "Overfitting occurs when a model learns noise.")
            self.assertTrue(data["has_sufficient_context"])
            self.assertEqual(len(data["sources"]), 1)
            self.assertEqual(data["sources"][0]["source_filename"], "ml.pdf")
            self.assertEqual(data["sources"][0]["page_number"], 3)
            self.assertEqual(data["sources"][0]["chunk_id"], "ml_p3_c0")
            self.assertEqual(data["sources"][0]["text"], "Overfitting occurs when a model learns noise.")

    @patch("app.api.chat.generate_answer")
    @patch("app.api.chat.retrieve")
    @patch("app.api.chat.load_bm25_index")
    @patch("app.api.chat.load_vector_store")
    def test_chat_returns_503_on_generation_unavailable_error(
        self, mock_load, mock_bm25, mock_retrieve, mock_generate
    ):
        """When Gemini generation raises GenerationUnavailableError, API returns HTTP 503 with detail."""
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()
        (self.vectorstore_dir / "bm25.pkl").touch()

        mock_load.return_value = MagicMock()
        mock_bm25.return_value = MagicMock()
        mock_retrieve.return_value = [MagicMock()]
        mock_generate.side_effect = GenerationUnavailableError(
            "The AI model is temporarily unavailable. Please try again shortly."
        )

        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.post("/api/chat", json={"query": "Explain the framework"})
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertEqual(
                data["detail"],
                "The AI model is temporarily unavailable. Please try again shortly.",
            )

    @patch("app.api.chat.generate_answer")
    @patch("app.api.chat.retrieve")
    @patch("app.api.chat.load_bm25_index")
    @patch("app.api.chat.load_vector_store")
    def test_chat_returns_429_on_generation_quota_exceeded_error(
        self, mock_load, mock_bm25, mock_retrieve, mock_generate
    ):
        """When Gemini generation raises GenerationQuotaExceededError, API returns HTTP 429 with detail."""
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()
        (self.vectorstore_dir / "bm25.pkl").touch()

        mock_load.return_value = MagicMock()
        mock_bm25.return_value = MagicMock()
        mock_retrieve.return_value = [MagicMock()]
        mock_generate.side_effect = GenerationQuotaExceededError(
            "Gemini API quota exhausted. Please try again after the quota resets."
        )

        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.post("/api/chat", json={"query": "Explain the framework"})
            self.assertEqual(response.status_code, 429)
            data = response.json()
            self.assertEqual(
                data["detail"],
                "Gemini API quota exhausted. Please try again after the quota resets.",
            )

    def test_chat_rejects_empty_query(self):
        """POST /api/chat with empty or whitespace-only query must return 400 or 422."""
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()

        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            # Empty string
            res1 = self.client.post("/api/chat", json={"query": ""})
            self.assertIn(res1.status_code, [400, 422])

            # Whitespace string
            res2 = self.client.post("/api/chat", json={"query": "   \n\t  "})
            self.assertIn(res2.status_code, [400, 422])

    def test_documents_list(self):
        """GET /api/documents must return all stored PDFs sorted deterministically."""
        (self.uploads_dir / "b_document.pdf").touch()
        (self.uploads_dir / "a_document.pdf").touch()

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.get("/api/documents")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("documents", data)
            filenames = [d["filename"] for d in data["documents"]]
            self.assertEqual(filenames, ["a_document.pdf", "b_document.pdf"])

    def test_upload_rejects_non_pdf(self):
        """POST /api/documents/upload must reject non-PDF files with HTTP 400."""
        # Non-PDF extension
        txt_file = io.BytesIO(b"Hello text file")
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", txt_file, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("pdf", response.json()["detail"].lower())

        # PDF extension but invalid header signature
        fake_pdf = io.BytesIO(b"Not a real PDF content")
        response2 = self.client.post(
            "/api/documents/upload",
            files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        )
        self.assertEqual(response2.status_code, 400)
        self.assertIn("invalid pdf", response2.json()["detail"].lower())

    @patch("app.api.documents.rebuild_vector_store")
    @patch("app.api.documents.load_pdf")
    @patch("app.api.documents.chunk_pages")
    def test_upload_success(self, mock_chunk, mock_load, mock_rebuild):
        """Valid PDF upload saves the file, invokes pipeline, and returns metadata."""
        mock_load.return_value = [
            PageDocument(
                source_filename="test.pdf",
                page_number=1,
                total_pages=1,
                text="Sample text",
                char_count=11,
                is_empty=False,
                doc_id="id123",
            )
        ]
        mock_chunk.return_value = [MagicMock()]

        pdf_content = b"%PDF-1.4\n%dummy pdf content\n%%EOF"
        pdf_file = io.BytesIO(pdf_content)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.post(
                "/api/documents/upload",
                files={"file": ("research.pdf", pdf_file, "application/pdf")},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["filename"], "research.pdf")
            self.assertEqual(data["pages"], 1)
            self.assertEqual(data["chunks"], 1)
            self.assertTrue((self.uploads_dir / "research.pdf").exists())
            mock_rebuild.assert_called_once()

    @patch("app.api.documents.rebuild_vector_store")
    @patch("app.api.documents.load_pdf")
    @patch("app.api.documents.chunk_pages")
    def test_upload_rejects_path_traversal(self, mock_chunk, mock_load, mock_rebuild):
        """Path traversal in filename is sanitized to prevent saving outside uploads directory."""
        mock_load.return_value = [MagicMock()]
        mock_chunk.return_value = [MagicMock()]

        pdf_content = b"%PDF-1.4\n%dummy pdf content\n%%EOF"
        pdf_file = io.BytesIO(pdf_content)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.post(
                "/api/documents/upload",
                files={"file": ("../../evil.pdf", pdf_file, "application/pdf")},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["filename"], "evil.pdf")
            # Must be saved strictly inside self.uploads_dir, not outside
            self.assertTrue((self.uploads_dir / "evil.pdf").exists())
            self.assertFalse((self.temp_path / "evil.pdf").exists())

    @patch("app.api.chat.generate_answer")
    @patch("app.api.chat.retrieve")
    @patch("app.api.chat.load_bm25_index")
    @patch("app.api.chat.load_vector_store")
    def test_chat_response_shape(self, mock_load, mock_bm25, mock_retrieve, mock_generate):
        """Chat API response must only contain answer, sources, and has_sufficient_context."""
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()
        (self.vectorstore_dir / "bm25.pkl").touch()

        mock_load.return_value = MagicMock()
        mock_bm25.return_value = MagicMock()
        mock_retrieve.return_value = []
        mock_generate.return_value = GeneratedAnswer(
            answer="Fallback answer.",
            sources=[],
            has_sufficient_context=False,
        )

        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.post("/api/chat", json={"query": "Question?"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data.keys()), {"answer", "sources", "has_sufficient_context"})

    def test_cors_configuration(self):
        """CORS middleware must allow the configured Vite origin (http://localhost:5173)."""
        response = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_cors_allows_delete_method(self):
        """CORS middleware must allow DELETE method from configured origin."""
        response = self.client.options(
            "/api/documents/test.pdf",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        self.assertEqual(response.status_code, 200)
        allow_methods = response.headers.get("access-control-allow-methods", "")
        self.assertIn("DELETE", allow_methods)

    @patch("app.api.documents.rebuild_vector_store")
    def test_delete_document_success(self, mock_rebuild):
        """Deleting an existing PDF removes it from storage, rebuilds index, and returns remaining count."""
        (self.uploads_dir / "doc1.pdf").touch()
        (self.uploads_dir / "doc2.pdf").touch()

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.delete("/api/documents/doc1.pdf")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["filename"], "doc1.pdf")
            self.assertEqual(data["remaining_documents"], 1)
            self.assertEqual(data["message"], "Document deleted successfully")
            self.assertFalse((self.uploads_dir / "doc1.pdf").exists())
            self.assertTrue((self.uploads_dir / "doc2.pdf").exists())
            mock_rebuild.assert_called_once()

    def test_delete_last_document_clears_indices(self):
        """Deleting the last remaining document clears FAISS & BM25 index files and causes /api/chat to 400."""
        # Put 1 document and index files in place
        (self.uploads_dir / "only.pdf").touch()
        (self.vectorstore_dir / "index.faiss").touch()
        (self.vectorstore_dir / "index.pkl").touch()
        (self.vectorstore_dir / "bm25.pkl").touch()

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.delete("/api/documents/only.pdf")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["filename"], "only.pdf")
            self.assertEqual(data["remaining_documents"], 0)

            # Assert index files were cleaned up
            self.assertFalse((self.vectorstore_dir / "index.faiss").exists())
            self.assertFalse((self.vectorstore_dir / "index.pkl").exists())
            self.assertFalse((self.vectorstore_dir / "bm25.pkl").exists())

        # Verify chat endpoint returns 400 when index is cleared
        with patch("app.api.chat.VECTORSTORE_DIR", self.vectorstore_dir):
            chat_res = self.client.post("/api/chat", json={"query": "Any query?"})
            self.assertEqual(chat_res.status_code, 400)
            self.assertIn("no documents are indexed", chat_res.json()["detail"].lower())

    def test_delete_nonexistent_document_returns_404(self):
        """Deleting a non-existent document returns HTTP 404."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.delete("/api/documents/nonexistent.pdf")
            self.assertEqual(response.status_code, 404)
            self.assertIn("not found", response.json()["detail"].lower())

    def test_delete_rejects_non_pdf(self):
        """Deleting a non-PDF filename returns HTTP 400."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.delete("/api/documents/malicious.sh")
            self.assertEqual(response.status_code, 400)
            self.assertIn("only pdf", response.json()["detail"].lower())

    def test_delete_rejects_path_traversal(self):
        """Path traversal attempts in delete endpoint return HTTP 400."""
        from fastapi import HTTPException
        from app.api.documents import delete_document

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            # Backslash traversal attempt in URL
            res1 = self.client.delete("/api/documents/..%5Cpasswords.pdf")
            self.assertEqual(res1.status_code, 400)
            self.assertIn("invalid", res1.json()["detail"].lower())

            # Dot-dot traversal attempt in URL
            res2 = self.client.delete("/api/documents/%2e%2e")
            self.assertEqual(res2.status_code, 400)
            self.assertIn("invalid", res2.json()["detail"].lower())

            # Direct function call with path traversal
            with self.assertRaises(HTTPException) as ctx:
                delete_document("../../etc/passwords.pdf")
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("invalid", ctx.exception.detail.lower())

    def _create_test_pdf(self, path: Path, text: str = "Sample document text") -> bytes:
        """Helper to create a valid 1-page PDF file with specified text."""
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        path.write_bytes(pdf_bytes)
        return pdf_bytes

    def test_get_document_file_without_chunk_id(self):
        """GET /api/documents/{filename}/file without chunk_id serves original PDF."""
        pdf_path = self.uploads_dir / "whitepaper.pdf"
        self._create_test_pdf(pdf_path, "System architecture overview")

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.get("/api/documents/whitepaper.pdf/file")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "application/pdf")
            self.assertIn("inline", response.headers.get("content-disposition", ""))
            self.assertIn("whitepaper.pdf", response.headers.get("content-disposition", ""))
            self.assertEqual(response.content, pdf_path.read_bytes())

    def test_get_document_file_with_valid_chunk_id_highlights_in_memory(self):
        """GET /api/documents/{filename}/file with valid chunk_id highlights text without altering disk file."""
        import pickle
        import pymupdf
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "01_Project_Atlas.pdf"
        evidence_text = "Project Atlas achieved recall@5 = 0.82."
        original_bytes = self._create_test_pdf(pdf_path, evidence_text)

        # Mock BM25 index with authoritative chunk metadata
        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "atlas_p1_c0",
                    "doc_id": "doc123",
                    "source_filename": "01_Project_Atlas.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": evidence_text,
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/01_Project_Atlas.pdf/file",
                params={"chunk_id": "atlas_p1_c0"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "application/pdf")

            # Check that returned PDF has a highlight annotation
            annotated_doc = pymupdf.open(stream=response.content, filetype="pdf")
            annots = list(annotated_doc[0].annots())
            self.assertEqual(len(annots), 1)
            annotated_doc.close()

            # Confirm original PDF on disk was never altered
            self.assertEqual(pdf_path.read_bytes(), original_bytes)

    def test_get_document_file_missing_pdf_returns_404(self):
        """GET /api/documents/{filename}/file returns 404 when file does not exist."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.get("/api/documents/ghost.pdf/file")
            self.assertEqual(response.status_code, 404)
            self.assertIn("not found", response.json()["detail"].lower())

    def test_get_document_file_path_traversal_rejected(self):
        """Path traversal attempts in file endpoint return HTTP 400."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            res1 = self.client.get("/api/documents/..%5Cpasswords.pdf/file")
            self.assertEqual(res1.status_code, 400)
            self.assertIn("invalid", res1.json()["detail"].lower())

    def test_get_document_file_nonexistent_chunk_id_returns_404(self):
        """Supplying a nonexistent chunk_id returns 404."""
        pdf_path = self.uploads_dir / "doc.pdf"
        self._create_test_pdf(pdf_path, "Content")

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/doc.pdf/file",
                params={"chunk_id": "nonexistent_c99"},
            )
            self.assertEqual(response.status_code, 404)
            self.assertIn("chunk not found", response.json()["detail"].lower())

    def test_get_document_file_chunk_document_mismatch_returns_400(self):
        """Supplying a chunk_id that belongs to another document returns 400."""
        import pickle
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "target.pdf"
        self._create_test_pdf(pdf_path, "Target content")

        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "other_p1_c0",
                    "doc_id": "doc456",
                    "source_filename": "other.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": "Other document content",
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/target.pdf/file",
                params={"chunk_id": "other_p1_c0"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("belong", response.json()["detail"].lower())

    def test_get_document_file_unsearchable_chunk_returns_unannotated_pdf(self):
        """When chunk text cannot be located on the page, endpoint still returns valid PDF."""
        import pickle
        import pymupdf
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "unsearchable.pdf"
        self._create_test_pdf(pdf_path, "Real page text")

        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "unsearch_p1_c0",
                    "doc_id": "doc789",
                    "source_filename": "unsearchable.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": "Completely different text that does not exist in the PDF",
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/unsearchable.pdf/file",
                params={"chunk_id": "unsearch_p1_c0"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "application/pdf")

            # Check that PDF has no highlight annotations
            doc = pymupdf.open(stream=response.content, filetype="pdf")
            self.assertIsNone(doc[0].first_annot)
            doc.close()

    def test_get_document_page_success_with_highlight_coordinates(self):
        """GET /api/documents/{filename}/page returns vector SVG and precise highlight coordinates without altering disk."""
        import pickle
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "01_Project_Atlas.pdf"
        full_text = (
            "Architecture\n\n"
            "In an internal experiment, Atlas achieved a retrieval recall@5 of 0.82 on the baseline question set.\n\n"
            "Evaluation"
        )
        original_bytes = self._create_test_pdf(pdf_path, full_text)

        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "atlas_p1_c0",
                    "doc_id": "doc123",
                    "source_filename": "01_Project_Atlas.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": full_text,
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/01_Project_Atlas.pdf/page",
                params={
                    "page_number": 1,
                    "chunk_id": "atlas_p1_c0",
                    "query": "What was the recall@5 achieved by Project Atlas?",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["filename"], "01_Project_Atlas.pdf")
            self.assertEqual(data["page_number"], 1)
            self.assertEqual(data["total_pages"], 1)
            self.assertGreater(data["page_width"], 0)
            self.assertGreater(data["page_height"], 0)
            self.assertTrue(data["svg"].startswith("<svg"))
            self.assertIn("Atlas achieved a retrieval recall@5 of 0.82", data["evidence_text"])
            self.assertNotIn("Architecture", data["evidence_text"])
            self.assertGreater(len(data["highlights"]), 0)
            for h in data["highlights"]:
                self.assertIn("x", h)
                self.assertIn("y", h)
                self.assertIn("width", h)
                self.assertIn("height", h)

            # Confirm original disk file remains completely untouched
            self.assertEqual(pdf_path.read_bytes(), original_bytes)

    def test_get_document_page_missing_pdf_returns_404(self):
        """GET /api/documents/{filename}/page returns 404 if file does not exist."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.get("/api/documents/nonexistent.pdf/page")
            self.assertEqual(response.status_code, 404)
            self.assertIn("not found", response.json()["detail"].lower())

    def test_get_document_page_invalid_page_number_returns_400(self):
        """GET /api/documents/{filename}/page returns 400 for out-of-bounds page numbers."""
        pdf_path = self.uploads_dir / "sample.pdf"
        self._create_test_pdf(pdf_path, "Sample page")

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            res_zero = self.client.get("/api/documents/sample.pdf/page", params={"page_number": 0})
            self.assertEqual(res_zero.status_code, 400)
            self.assertIn("invalid page number", res_zero.json()["detail"].lower())

            res_high = self.client.get("/api/documents/sample.pdf/page", params={"page_number": 99})
            self.assertEqual(res_high.status_code, 400)
            self.assertIn("invalid page number", res_high.json()["detail"].lower())

    def test_get_document_page_path_traversal_rejected(self):
        """Path traversal attempts in page endpoint return HTTP 400."""
        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir):
            response = self.client.get("/api/documents/..%5Cpasswords.pdf/page")
            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid", response.json()["detail"].lower())

    def test_get_document_page_chunk_document_mismatch_returns_400(self):
        """Supplying a chunk_id belonging to a different document returns 400."""
        import pickle
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "doc_a.pdf"
        self._create_test_pdf(pdf_path, "Doc A text")

        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "chunk_b",
                    "doc_id": "doc_b",
                    "source_filename": "doc_b.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": "Doc B text",
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/doc_a.pdf/page",
                params={"chunk_id": "chunk_b"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("belong", response.json()["detail"].lower())

    def test_get_document_page_nonexistent_chunk_returns_404(self):
        """Supplying an unknown chunk_id returns 404."""
        pdf_path = self.uploads_dir / "doc.pdf"
        self._create_test_pdf(pdf_path, "Content")

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/doc.pdf/page",
                params={"chunk_id": "unknown_chunk_id"},
            )
            self.assertEqual(response.status_code, 404)
            self.assertIn("chunk not found", response.json()["detail"].lower())

    def test_atlas_highlight_y_coordinate_and_svg_render(self):
        """Atlas recall@5 query highlight has y-coordinate corresponding to Evaluation text, not intro paragraph."""
        import pickle
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "01_Project_Atlas.pdf"
        full_text = (
            "Project Atlas\n\n"
            "Project Atlas is a document retrieval platform designed for research teams.\n\n"
            "Architecture\n\n"
            "Atlas uses a hybrid retrieval pipeline combining semantic embeddings and sparse lexical matching.\n\n"
            "Evaluation\n\n"
            "In an internal experiment, Atlas achieved a retrieval recall@5 of 0.82 on the baseline question set.\n"
            "The team considers recall@5 the primary retrieval metric for this experiment."
        )
        self._create_test_pdf(pdf_path, full_text)

        evidence_sentence = "In an internal experiment, Atlas achieved a retrieval recall@5 of 0.82 on the baseline question set."
        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "atlas_chunk_0",
                    "doc_id": "atlas_doc",
                    "source_filename": "01_Project_Atlas.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": full_text,
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/01_Project_Atlas.pdf/page",
                params={
                    "page_number": 1,
                    "chunk_id": "atlas_chunk_0",
                    "query": "What was the recall@5 achieved by Project Atlas?",
                    "evidence_text": evidence_sentence,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["evidence_text"], evidence_sentence)
            self.assertGreater(len(data["highlights"]), 0)

            # Check that the highlight rectangle corresponds to Evaluation section (y > 200),
            # NOT the introductory Project Atlas paragraph (y ~ 105)
            atlas_h = data["highlights"][0]
            self.assertGreater(atlas_h["y"], 200.0)
            self.assertNotIn("Project Atlas is a document retrieval platform", data["evidence_text"])

            # Verify that the generated SVG itself contains the drawn highlight
            self.assertIn("fill=\"#ffa600\"", data["svg"])
            self.assertIn("fill-opacity=\".3\"", data["svg"])

    def test_rrf_highlight_in_architecture_section(self):
        """RRF query highlight covers Reciprocal Rank Fusion in the Architecture section."""
        import pickle
        from app.rag.bm25_store import BM25Index

        pdf_path = self.uploads_dir / "01_Project_Atlas.pdf"
        full_text = (
            "Project Atlas\n\n"
            "Project Atlas is a document retrieval platform designed for research teams.\n\n"
            "Architecture\n\n"
            "Atlas uses a hybrid retrieval pipeline. It combines dense semantic retrieval with BM25 lexical "
            "retrieval and then combines their rankings using Reciprocal Rank Fusion (RRF).\n\n"
            "Evaluation\n\n"
            "In an internal experiment, Atlas achieved a retrieval recall@5 of 0.82 on the baseline question set."
        )
        self._create_test_pdf(pdf_path, full_text)

        bm25_data = BM25Index(
            bm25=None,
            chunks=[
                {
                    "chunk_id": "atlas_chunk_0",
                    "doc_id": "atlas_doc",
                    "source_filename": "01_Project_Atlas.pdf",
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": 0,
                    "text": full_text,
                }
            ],
        )
        with open(self.vectorstore_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        with patch("app.api.documents.UPLOADS_DIR", self.uploads_dir), \
             patch("app.api.documents.VECTORSTORE_DIR", self.vectorstore_dir):
            response = self.client.get(
                "/api/documents/01_Project_Atlas.pdf/page",
                params={
                    "page_number": 1,
                    "chunk_id": "atlas_chunk_0",
                    "query": "What does RRF stand for?",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("Reciprocal Rank Fusion (RRF)", data["evidence_text"])
            self.assertGreater(len(data["highlights"]), 0)

            # RRF is in Architecture section (y between 120 and 220)
            for h in data["highlights"]:
                self.assertGreaterEqual(h["y"], 120.0)
                self.assertLess(h["y"], 240.0)

            # Generated SVG contains the highlight
            self.assertIn("fill=\"#ffa600\"", data["svg"])


if __name__ == "__main__":
    unittest.main()
