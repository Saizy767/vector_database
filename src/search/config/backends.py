from pydantic import BaseModel, Field

class PGVectorConfig(BaseModel):
    db_url: str = Field(..., description="PostgreSQL URL с pgvector")
