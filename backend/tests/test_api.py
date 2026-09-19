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
            sources=[SourceCitation(source_filename="ml.pdf", page_number=3)],
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


if __name__ == "__main__":
    unittest.main()
