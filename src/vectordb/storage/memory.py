import time
from typing import Dict, List, Optional, Callable
from vectordb.models import DocumentMetadata, DocumentEmbedding
from vectordb.storage.interface import StorageInterface


class MemoryStorage(StorageInterface):
    def __init__(self):
        self._metadatas: Dict[str, DocumentMetadata] = {}
        self._embeddings: Dict[str, DocumentEmbedding] = {}


    # METADATA
    def upsert_metadata(self, meta: DocumentMetadata):
        if meta.id in self._metadatas:
            old = self._metadatas[meta.id]
            meta.version = old.version + 1
            meta.created_at = old.created_at
            meta.updated_at = time.time()
        self._metadatas[meta.id] = meta


    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        meta = self._metadatas.get(doc_id)
        if meta and not meta.is_deleted:
            return meta
        return None


    def delete_metadata(self, doc_id: str):
        if doc_id in self._metadatas:
            del self._metadatas[doc_id]


    def query_metadata_all(self) -> List[DocumentMetadata]:
        return list(self._metadatas.values())


    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        return [m for m in self._metadatas.values() if not m.is_deleted and predicate(m.metadata)]


    # EMBEDDINGS
    def upsert_embedding(self, emb: DocumentEmbedding):
        if emb.id in self._embeddings:
            old = self._embeddings[emb.id]
            emb.version = old.version + 1
            emb.created_at = old.created_at
            emb.updated_at = time.time()
        self._embeddings[emb.id] = emb


    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        emb = self._embeddings.get(doc_id)
        if emb and not emb.is_deleted:
            return emb.embedding
        return None


    def delete_embedding(self, doc_id: str):
        if doc_id in self._embeddings:
            del self._embeddings[doc_id]


    def all_ids_with_embeddings(self) -> List[str]:
        return [eid for eid, emb in self._embeddings.items() if not emb.is_deleted]


    def delete(self, doc_id: str, hard: bool = False):
        if hard:
            self._metadatas.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)
        else:
            if doc_id in self._metadatas:
                self._metadatas[doc_id].is_deleted = True
                self._metadatas[doc_id].updated_at = time.time()
            if doc_id in self._embeddings:
                self._embeddings[doc_id].is_deleted = True
                self._embeddings[doc_id].updated_at = time.time()

