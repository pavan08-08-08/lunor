import os
from pathlib import Path

# Base project directory resolved relative to backend/app/config.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories with environment variable overrides for testing
DATA_DIR = Path(os.environ.get("LUNOR_DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = Path(os.environ.get("LUNOR_UPLOADS_DIR", DATA_DIR / "uploads"))
VECTORSTORE_DIR = Path(os.environ.get("LUNOR_VECTORSTORE_DIR", DATA_DIR / "vectorstore"))
DOCUMENTS_DIR = Path(os.environ.get("LUNOR_DOCUMENTS_DIR", DATA_DIR / "documents"))


def ensure_directories() -> None:
    """Ensure essential data directories exist on the filesystem."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
