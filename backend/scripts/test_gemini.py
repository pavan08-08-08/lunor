"""Manual integration smoke test for Gemini grounded generation.

This script only runs when GEMINI_API_KEY is configured in the environment or .env.
It executes the end-to-end pipeline:
PDF -> loader -> chunker -> embeddings -> FAISS -> retrieve -> generate_answer
"""

import os
from pathlib import Path
import sys

# Ensure backend package is in python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv

# Load .env
load_dotenv()
backend_env = backend_dir / ".env"
if backend_env.exists():
    load_dotenv(dotenv_path=backend_env)

from app.rag.chunker import chunk_pages
from app.rag.embeddings import embed_chunks
from app.rag.generator import generate_answer
from app.rag.loader import load_pdf
from app.rag.retriever import retrieve
from app.rag.vector_store import build_vector_store


def run_gemini_smoke_test():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print("GEMINI_API_KEY is not configured. Skipping real Gemini integration test.")
        print("To run this test, set GEMINI_API_KEY in your environment or backend/.env file.")
        return False

    pdf_path = Path(__file__).resolve().parent.parent.parent / "data" / "documents" / "sample1_paper copy.pdf"
    if not pdf_path.exists():
        print(f"Sample document not found at {pdf_path}")
        return False

    question = "What is the primary methodology proposed in this paper for traffic risk prediction?"

    print("=== RUNNING REAL GEMINI SMOKE TEST ===")
    print("1. Loading PDF...")
    pages = load_pdf(str(pdf_path))
    print(f"   Loaded {len(pages)} pages.")

    print("2. Chunking pages...")
    chunks = chunk_pages(pages)
    print(f"   Produced {len(chunks)} chunks.")

    print("3. Generating embeddings...")
    embedded = embed_chunks(chunks)
    print(f"   Generated {len(embedded)} vectors.")

    print("4. Building FAISS index...")
    store = build_vector_store(embedded)
    print("   FAISS vector store built.")

    print(f"5. Retrieving context for query: '{question}'...")
    retrieved = retrieve(question, store, k=3, threshold=0.35)
    print(f"   Retrieved {len(retrieved)} relevant chunks.")

    print("6. Calling Gemini for grounded answer generation...")
    result = generate_answer(question, retrieved)

    print("\n=== GEMINI GENERATION RESULT ===")
    print(f"Question:\n{question}\n")
    print(f"Generated Answer:\n{result.answer}\n")
    print("Sources:")
    for src in result.sources:
        print(f"- {src.source_filename}, page {src.page_number}")
    print(f"Has Sufficient Context: {result.has_sufficient_context}")

    return True


if __name__ == "__main__":
    run_gemini_smoke_test()
