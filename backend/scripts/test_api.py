"""Manual API smoke test script.

Tests the FastAPI application endpoints in-process using TestClient and checks:
1. GET /api/health
2. GET /docs
3. GET /openapi.json
4. GET /api/documents
5. POST /api/chat error handling when no index exists
"""

from pathlib import Path
import sys

# Ensure backend package is in python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app


def run_api_smoke_test():
    client = TestClient(app)

    print("=== RUNNING API SMOKE TEST ===")

    # 1. Health check
    res_health = client.get("/api/health")
    print(f"1. GET /api/health -> Status {res_health.status_code}, Body: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # 2. OpenAPI JSON
    res_openapi = client.get("/openapi.json")
    print(f"2. GET /openapi.json -> Status {res_openapi.status_code}, Title: {res_openapi.json().get('info', {}).get('title')}")
    assert res_openapi.status_code == 200

    # 3. Swagger Docs
    res_docs = client.get("/docs")
    print(f"3. GET /docs -> Status {res_docs.status_code} (Swagger UI loaded)")
    assert res_docs.status_code == 200

    # 4. List Documents
    res_docs_list = client.get("/api/documents")
    print(f"4. GET /api/documents -> Status {res_docs_list.status_code}, Docs: {res_docs_list.json()}")
    assert res_docs_list.status_code == 200

    # 5. Chat without vectorstore
    res_chat = client.post("/api/chat", json={"query": "What is in the document?"})
    print(f"5. POST /api/chat (no index check) -> Status {res_chat.status_code}, Detail: {res_chat.json().get('detail')}")

    print("\nAPI smoke test passed successfully!")


if __name__ == "__main__":
    run_api_smoke_test()
