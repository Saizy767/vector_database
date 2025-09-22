from __future__ import annotations
from typing import List

class EmbeddingProvider:
    """Abstract embedding provider.
    Implementations must provide embed_texts and embed_text.
    """
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError