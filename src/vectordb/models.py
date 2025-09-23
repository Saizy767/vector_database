from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import time


@dataclass
class DocumentMetadata:
    id: str                     # hash of embedding (or user-defined id)
    metadata: Dict[str, Any]     # arbitrary metadata
    text: Optional[str] = None   # optional document text
    created_at: float = field(default_factory=lambda: time.time())  # creation timestamp
    updated_at: float = field(default_factory=lambda: time.time())  # last update timestamp
    version: int = 1             # version of the document
    is_deleted: bool = False     # soft delete flag
    similar_ids: List[str] = field(default_factory=list)  # list of similar document ids

@dataclass
class DocumentEmbedding:
    id: str                     # hash of embedding (matches metadata.id)
    embedding: List[float]      # normalized embedding vector
    is_deleted: bool = False    # soft delete flag for embedding