import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    db_url: str = Field(default="", env="DB_URL")
    test_db_url: Optional[str] = None
    embedding_provider: str = Field(default="sentence-transformer", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="dummy", env="EMBEDDING_MODEL")
    device: str = Field(default="cpu", env="DEVICE")
    
    metadata_columns: List[str] = Field(default_factory=list, env="METADATA_COLUMNS")
    embedding_columns: List[str] = Field(default_factory=list, env="EMBEDDING_COLUMNS")
    
    extract_table_name: str = Field(..., env="EXTRACT_TABLE_NAME")
    load_table_name: str = Field(..., env="LOAD_TABLE_NAME")
    source_id: str = Field(..., env="SOURCE_ID")

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8"
    )

    @field_validator("db_url")
    def check_db_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("❌ DB_URL is missing or empty in .env")
        return v.strip()

    @field_validator("metadata_columns", "embedding_columns", mode="before")
    def parse_list_fields(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Убираем возможные кавычки и скобки из строки (защита от ошибок в .env)
            v = v.strip().strip('"').strip("'")
            if v.startswith("[") and v.endswith("]"):
                # Попытка распарсить как JSON-массив (опционально)
                import json
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        return parsed
                except (ValueError, TypeError):
                    pass
            # Иначе — обрабатываем как CSV
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return v
        return []


settings = Settings()