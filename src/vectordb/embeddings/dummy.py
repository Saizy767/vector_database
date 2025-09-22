from .base import EmbeddingProvider
from typing import List
import math

class DummyEmbedding(EmbeddingProvider):
    """Deterministic dummy embeddings for testing.
    Not suitable for production; returns a fixed-length vector derived from characters.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text[: self.dim]):
            vec[i] = (ord(ch) % 97) / 97.0
        # simple non-zero distribution
        if all(v == 0.0 for v in vec):
            vec[0] = 1.0
        # normalize
        s = math.sqrt(sum(x * x for x in vec))
        return [x / s for x in vec]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_text(self, text: str) -> List[float]:
        return self._embed(text)