from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    db_url: str = Field(default="", env="DB_URL")

    embedding_provider:str = Field(default="sentense-transformer", env="EMBEDDING_PROVIDER")
    embedding_model:str = Field(default="dummy", env="EMBEDDING_MODEL")
    device:str = Field(default="cpu", env="DEVICE")

    class Config:
        env_files = '.env'
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()