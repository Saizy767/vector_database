import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent.parent.parent

class SearchSettings(BaseSettings):
    search_backend: Literal["pgvector"] = Field("pgvector", env="SEARCH_BACKEND")
    db_url: str = Field(..., env="DB_URL")
    top_k: int = Field(10, ge=1, le=100, env="SEARCH_TOP_K")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, env="SEARCH_MIN_SIMILARITY")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    device: str = Field("cpu", env="DEVICE")
    load_table_name: str = Field(..., env="LOAD_TABLE_NAME")

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_backend_config(self) -> dict:
        if self.search_backend == "pgvector":
            return {"db_url": self.db_url}
        else:
            raise ValueError(f"Unsupported backend: {self.search_backend}")

settings = SearchSettings()