from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    text: str
    top_k: int = Field(10, ge=1, le=100)
    min_similarity: float = Field(0.7, ge=0.0, le=1.0)
    metadata_filter: Optional[Dict[str, Any]] = None

class SearchResultItem(BaseModel):
    id: int
    chunk_text: str
    metadata: Dict[str, Any]
    similarity: float

class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int