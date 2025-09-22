from typing import List, Tuple
import numpy as np

class BruteForceIndex:
    """Simple brute-force vector index using cosine similarity"""
    def __init__(self):
        self._vectors = {}  # dict of doc_id -> embedding

    def add(self, doc_id: str, embedding: List[float]):
        """Add a new embedding to the index"""
        self._vectors[doc_id] = np.array(embedding, dtype=np.float32)

    def update(self, doc_id: str, embedding: List[float]):
        """Update embedding in the index"""
        self._vectors[doc_id] = np.array(embedding, dtype=np.float32)

    def remove(self, doc_id: str):
        """Remove embedding from the index"""
        if doc_id in self._vectors:
            del self._vectors[doc_id]

    def search(self, query: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k closest doc_ids by cosine similarity"""
        if not self._vectors:
            return []
        query_vec = np.array(query, dtype=np.float32)
        doc_ids = list(self._vectors.keys())
        vectors = np.stack([self._vectors[doc_id] for doc_id in doc_ids])
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-10
        sims = vectors @ query_vec / norms
        # Get top-k indices
        topk_idx = np.argsort(-sims)[:k]
        return [(doc_ids[i], float(sims[i])) for i in topk_idx]