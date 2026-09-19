import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = BACKEND_DIR.parent

# Robustly load environment configuration from backend/.env or root .env
backend_env = BACKEND_DIR / ".env"
root_env = BASE_DIR / ".env"
if backend_env.exists():
    load_dotenv(dotenv_path=backend_env)
elif root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

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
