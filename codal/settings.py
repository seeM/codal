import os
from pathlib import Path

_DEFAULT_XDG_CACHE_HOME = Path.home() / ".cache"
XDG_CACHE_HOME = Path(os.getenv("XDG_CACHE_HOME", _DEFAULT_XDG_CACHE_HOME)).expanduser()
CACHE_DIR = XDG_CACHE_HOME / "codal"
REPO_DIR = CACHE_DIR / "repos"
INDEX_DIR = CACHE_DIR / "indexes"
DB_PATH = CACHE_DIR / "db.sqlite"

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
COMPLETION_MODEL_NAME = "gpt-4"
