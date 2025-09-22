from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Callable
from vectordb.storage.interface import StorageInterface
from vectordb.index.brute_force import BruteForceIndex
from vectordb.models import DocumentMetadata
from vectordb.utils import normalize_vector

class VectorDB:
    def __init__(self, storage: StorageInterface, index=None, embedder=None):
        self.storage = storage
        self.index = index or BruteForceIndex()
        self.embedder = embedder
        self._rebuild_index()

    def _rebuild_index(self):
        try:
            ids = self.storage.all_ids_with_embeddings()
        except Exception:
            ids = []
        for doc_id in ids:
            emb = self.storage.get_embedding(doc_id)
            if emb is not None:
                self.index.add(doc_id, emb)

    def upsert(self, doc_id: str, embedding: Optional[List[float]] = None, metadata: Optional[dict] = None, text: Optional[str] = None):
        if embedding is None:
            if self.embedder is not None and text is not None:
                embedding = self.embedder.embed_text(text)
            else:
                raise RuntimeError('Embedding is required (either provide embedding or embedder+text)')
        emb_norm = normalize_vector(embedding)
        meta_obj = DocumentMetadata(id=doc_id, metadata=(metadata or {}), text=text)
        self.storage.upsert_metadata(meta_obj)
        self.storage.upsert_embedding(doc_id, emb_norm)
        self.index.update(doc_id, emb_norm)

    def get(self, doc_id: str) -> Optional[Tuple[DocumentMetadata, Optional[List[float]]]]:
        meta = self.storage.get_metadata(doc_id)
        emb = self.storage.get_embedding(doc_id)
        return (meta, emb)

    def delete(self, doc_id: str):
        self.storage.delete(doc_id)
        self.index.remove(doc_id)

    def update_metadata(self, doc_id: str, metadata_patch: Dict[str, Any]):
        meta = self.storage.get_metadata(doc_id)
        if meta is None:
            raise KeyError(f'Document {doc_id} metadata not found')
        meta.metadata.update(metadata_patch)
        self.storage.upsert_metadata(meta)

    def search(self, query_embedding: Optional[List[float]] = None, k: int = 10, filter_metadata: Optional[Callable[[dict], bool]] = None) -> List[Tuple[DocumentMetadata, float]]:
        if query_embedding is None:
            raise RuntimeError('query_embedding is required for search in this schema')
        q = normalize_vector(query_embedding)
        hits = self.index.search(q, k=k)
        results: List[Tuple[DocumentMetadata, float]] = []
        for doc_id, score in hits:
            meta = self.storage.get_metadata(doc_id)
            if meta is None:
                continue
            if filter_metadata and not filter_metadata(meta.metadata):
                continue
            results.append((meta, score))
        return results

    def query_by_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        return self.storage.query_metadata(predicate)