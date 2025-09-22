from .base import EmbeddingProvider
from vectordb.utils import normalize_vector

class DummyEmbedding(EmbeddingProvider):
    """Deterministic dummy embeddings for testing.
    Not suitable for production; returns a fixed-length vector derived from characters.
    """
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_text(self, text: str):
        """Convert text to a simple numeric vector of fixed dimension"""
        vec = [ord(c) % 10 for c in text[:self.dim]]
        # Pad to dim
        vec += [0] * (self.dim - len(vec))
        return normalize_vector(vec)