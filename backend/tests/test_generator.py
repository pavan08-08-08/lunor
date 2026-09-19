import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import errors

from app.rag.generator import (
    FALLBACK_ANSWER,
    GEMINI_MODEL,
    GeneratedAnswer,
    GenerationUnavailableError,
    SourceCitation,
    _format_context,
    _reset_gemini_client,
    generate_answer,
    get_gemini_client,
)
from app.rag.retriever import RetrievedChunk


def _make_retrieved_chunk(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    source_filename: str = "test.pdf",
    page_number: int = 1,
    total_pages: int = 5,
    chunk_index: int = 0,
    text: str = "Sample chunk content for generator testing.",
    similarity_score: float = 0.88,
) -> RetrievedChunk:
    """Helper to construct a RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_number=page_number,
        total_pages=total_pages,
        chunk_index=chunk_index,
        text=text,
        similarity_score=similarity_score,
    )


class TestGenerator(unittest.TestCase):
    def setUp(self):
        _reset_gemini_client()

    def tearDown(self):
        _reset_gemini_client()

    def test_empty_retrieval_does_not_call_gemini(self):
        """When retrieved_chunks is empty, return fallback answer without calling Gemini."""
        with patch("app.rag.generator.get_gemini_client") as mock_get_client:
            res = generate_answer("What is the refund policy?", [])
            self.assertEqual(res.answer, FALLBACK_ANSWER)
            self.assertEqual(res.sources, [])
            self.assertFalse(res.has_sufficient_context)
            mock_get_client.assert_not_called()

    @patch("app.rag.generator.get_gemini_client")
    def test_context_contains_chunk_text(self, mock_get_client):
        """Context passed to Gemini must contain the text of all retrieved chunks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a synthesized answer."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = [
            _make_retrieved_chunk(chunk_id="c1", text="Machine learning algorithms optimize flow."),
            _make_retrieved_chunk(chunk_id="c2", text="Traffic signals adjust adaptively to congestion."),
        ]

        generate_answer("How do signals adjust?", chunks)

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]

        self.assertIn("Machine learning algorithms optimize flow.", contents)
        self.assertIn("Traffic signals adjust adaptively to congestion.", contents)

    @patch("app.rag.generator.get_gemini_client")
    def test_context_contains_source_and_page_labels(self, mock_get_client):
        """Context passed to Gemini must contain formatted source and page metadata labels."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounding answer."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = [
            _make_retrieved_chunk(source_filename="whitepaper.pdf", page_number=3),
            _make_retrieved_chunk(source_filename="whitepaper.pdf", page_number=7),
        ]

        generate_answer("Summary request", chunks)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]

        self.assertIn("[Source 1: whitepaper.pdf, page 3]", contents)
        self.assertIn("[Source 2: whitepaper.pdf, page 7]", contents)

    @patch("app.rag.generator.get_gemini_client")
    def test_successful_generation(self, mock_get_client):
        """Successful generation returns answer text and has_sufficient_context=True."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Customers are eligible for refunds within 30 days of purchase."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunk = _make_retrieved_chunk(text="Refunds are granted within 30 days.")
        res = generate_answer("What is the refund window?", [chunk])

        self.assertIsInstance(res, GeneratedAnswer)
        self.assertEqual(res.answer, "Customers are eligible for refunds within 30 days of purchase.")
        self.assertTrue(res.has_sufficient_context)
        self.assertEqual(len(res.sources), 1)

    @patch("app.rag.generator.get_gemini_client")
    def test_sources_are_constructed_from_metadata(self, mock_get_client):
        """Source citations are constructed directly from chunk metadata, not LLM parsing."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated summary."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = [
            _make_retrieved_chunk(source_filename="doc_A.pdf", page_number=2),
            _make_retrieved_chunk(source_filename="doc_B.pdf", page_number=5),
        ]

        res = generate_answer("Explain both documents", chunks)

        self.assertEqual(
            res.sources,
            [
                SourceCitation(source_filename="doc_A.pdf", page_number=2),
                SourceCitation(source_filename="doc_B.pdf", page_number=5),
            ],
        )

    @patch("app.rag.generator.get_gemini_client")
    def test_duplicate_page_sources_are_deduplicated(self, mock_get_client):
        """Multiple chunks from the same (filename, page) are deduplicated while preserving order."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer from repeated page."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = [
            _make_retrieved_chunk(source_filename="paper.pdf", page_number=4, chunk_index=0),
            _make_retrieved_chunk(source_filename="paper.pdf", page_number=4, chunk_index=1),
            _make_retrieved_chunk(source_filename="paper.pdf", page_number=5, chunk_index=0),
            _make_retrieved_chunk(source_filename="paper.pdf", page_number=4, chunk_index=2),
        ]

        res = generate_answer("Question", chunks)

        self.assertEqual(
            res.sources,
            [
                SourceCitation(source_filename="paper.pdf", page_number=4),
                SourceCitation(source_filename="paper.pdf", page_number=5),
            ],
        )

    def test_client_singleton_reuse(self):
        """get_gemini_client() returns the same client instance across multiple calls."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key"}):
            client_1 = get_gemini_client()
            client_2 = get_gemini_client()
            self.assertIs(client_1, client_2)

    @patch("app.rag.generator.get_gemini_client")
    def test_query_is_passed_to_generation(self, mock_get_client):
        """The user's query must be included in the generation contents."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        query_text = "What specific optimization criteria were evaluated?"
        chunks = [_make_retrieved_chunk(text="Optimization criteria included RMSE and MAE.")]

        generate_answer(query_text, chunks)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        self.assertIn(query_text, contents)
        self.assertEqual(call_kwargs["model"], GEMINI_MODEL)

    @patch("app.rag.generator.get_gemini_client")
    def test_generated_text_is_returned_without_fake_citations(self, mock_get_client):
        """LLM response is returned as-is without attempt to parse or invent citation text."""
        fake_llm_text = "According to [Ref 123] and Dr. Smith (2024), urban congestion declined."
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = fake_llm_text
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunk = _make_retrieved_chunk(source_filename="urban.pdf", page_number=1)
        res = generate_answer("What happened?", [chunk])

        # Exact LLM text preserved in answer
        self.assertEqual(res.answer, fake_llm_text)
        # Citations are solely from chunk metadata, ignoring fake LLM citations
        self.assertEqual(res.sources, [SourceCitation(source_filename="urban.pdf", page_number=1)])

    def test_missing_api_key_has_clear_error(self):
        """get_gemini_client() raises ValueError when GEMINI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_gemini_client()
            self.assertIn("gemini_api_key", str(ctx.exception).lower())

    @patch("app.rag.generator.get_gemini_client")
    def test_empty_response_from_gemini_raises_runtime_error(self, mock_get_client):
        """When Gemini returns an empty or whitespace-only response, raise RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunk = _make_retrieved_chunk(text="Some text")
        with self.assertRaises(RuntimeError) as ctx:
            generate_answer("Query", [chunk])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_generation_unavailable_error_exists(self):
        """GenerationUnavailableError must exist and be an Exception subclass."""
        self.assertTrue(issubclass(GenerationUnavailableError, Exception))
        err = GenerationUnavailableError("Service temporary unavailable")
        self.assertEqual(str(err), "Service temporary unavailable")

    @patch("app.rag.generator.get_gemini_client")
    def test_server_error_translated_to_generation_unavailable_error(self, mock_get_client):
        """Terminal Gemini ServerError (e.g. 503) must be translated to GenerationUnavailableError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = errors.ServerError(
            503, {"error": {"message": "Model is experiencing high demand"}}
        )
        mock_get_client.return_value = mock_client

        chunk = _make_retrieved_chunk(text="Document text")
        with self.assertRaises(GenerationUnavailableError) as ctx:
            generate_answer("What is the framework?", [chunk])

        self.assertEqual(
            str(ctx.exception),
            "The AI model is temporarily unavailable. Please try again shortly.",
        )

    @patch("app.rag.generator.get_gemini_client")
    def test_client_error_not_converted_to_generation_unavailable_error(self, mock_get_client):
        """ClientError (e.g. 400 bad request) must propagate and not be converted to GenerationUnavailableError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = errors.ClientError(
            400, {"error": {"message": "Invalid argument"}}
        )
        mock_get_client.return_value = mock_client

        chunk = _make_retrieved_chunk(text="Document text")
        with self.assertRaises(errors.ClientError):
            generate_answer("Query", [chunk])

    def test_client_configured_with_native_retry_options(self):
        """get_gemini_client() configures Client with HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0, exp_base=2.0)."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_cls:
            _reset_gemini_client()
            get_gemini_client()

            mock_client_cls.assert_called_once()
            call_kwargs = mock_client_cls.call_args.kwargs
            self.assertEqual(call_kwargs["api_key"], "test-key")
            self.assertIn("http_options", call_kwargs)
            http_options = call_kwargs["http_options"]
            self.assertIsNotNone(http_options.retry_options)
            retry_opts = http_options.retry_options
            self.assertEqual(retry_opts.attempts, 3)
            self.assertEqual(retry_opts.initial_delay, 1.0)
            self.assertEqual(retry_opts.max_delay, 8.0)
            self.assertEqual(retry_opts.exp_base, 2.0)


if __name__ == "__main__":
    unittest.main()
