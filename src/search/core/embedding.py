from shared.embedding.base import BaseEmbedding
from .abc import BaseEmbeddingProvider

class SharedEmbeddingAdapter(BaseEmbeddingProvider):
    def __init__(self, embedder: BaseEmbedding):
        self.embedder = embedder

    def embed_text(self, text: str) -> list[float]:
        emb = self.embedder.embed_text(text)
        return emb.tolist() if hasattr(emb, 'tolist') else emb.tolist()