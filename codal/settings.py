from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    CACHE_DIR: Path = Path.home() / ".cache/codal"
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
    COMPLETION_MODEL_NAME: str = "gpt-4"

    @property
    def REPO_DIR(self) -> Path:
        return self.CACHE_DIR / "repos"

    @property
    def INDEX_DIR(self) -> Path:
        return self.CACHE_DIR / "indexes"

    @property
    def DB_PATH(self) -> Path:
        return self.CACHE_DIR / "db.sqlite"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"sqlite:///{self.DB_PATH}"

    class Config:
        env_prefix = "CODAL_"
        case_sensitive = True


settings = Settings()
