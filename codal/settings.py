from pathlib import Path

CACHE_DIR = Path.home() / ".cache/codal"
REPO_DIR = CACHE_DIR / "repos"
INDEX_DIR = CACHE_DIR / "indexes"
DB_PATH = CACHE_DIR / "db.sqlite"

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
COMPLETION_MODEL_NAME = "gpt-4"
