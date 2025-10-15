# src/search/config/settings.py

import os
from pathlib import Path
from typing import Literal, Union, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


from .backends import PGVectorConfig

BASE_DIR = Path(__file__).parent.parent.parent

class SearchSettings(BaseSettings):
    search_backend: Literal["pgvector"] = Field("pgvector", env="SEARCH_BACKEND")
    top_k: int = Field(10, ge=1, le=100, env="SEARCH_TOP_K")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, env="SEARCH_MIN_SIMILARITY")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    device: str = Field("cpu", env="DEVICE")

    pgvector: Optional[PGVectorConfig] = None

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_backend_config(self) -> Union[PGVectorConfig]:
        if self.search_backend == "pgvector":
            if not self.pgvector:
                raise ValueError("pgvector config is required when SEARCH_BACKEND=pgvector")
            return self.pgvector
        else:
            raise ValueError(f"Unsupported backend: {self.search_backend}")

settings = SearchSettings()