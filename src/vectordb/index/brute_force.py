from __future__ import annotations
from typing import List, Tuple
import numpy as np
from vectordb.models import DocumentMetadata
from vectordb.utils import cosine_similarity_matrix, normalize_vector

class BruteForceIndex:
    """Simple in-memory index that keeps embeddings and performs brute-force search."""

    def __init__(self):
        # mapping id -> embedding
        self._id_to_embedding: dict[str, List[float]] = {}

    def add(self, doc_id: str, embedding: List[float]):
        self._id_to_embedding[doc_id] = normalize_vector(embedding)

    def remove(self, doc_id: str):
        if doc_id in self._id_to_embedding:
            del self._id_to_embedding[doc_id]

    def update(self, doc_id: str, embedding: List[float]):
        self.add(doc_id, embedding)

    def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        ids = list(self._id_to_embedding.keys())
        if not ids:
            return []
        vectors = [self._id_to_embedding[i] for i in ids]
        sims = cosine_similarity_matrix(query_embedding, vectors)
        # pair and sort
        paired = list(zip(ids, sims))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:k]

    def all_ids(self) -> List[str]:
        return list(self._id_to_embedding.keys())