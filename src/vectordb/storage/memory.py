from __future__ import annotations
from typing import Dict, List, Optional, Callable
from vectordb.models import DocumentMetadata
from vectordb.storage.interface import StorageInterface


class MemoryStorage(StorageInterface):
    def __init__(self):
        self._metadatas: Dict[str, DocumentMetadata] = {}
        self._embeddings: Dict[str, List[float]] = {}


    # METADATA
    def upsert_metadata(self, meta: DocumentMetadata):
        self._metadatas[meta.id] = meta


    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self._metadatas.get(doc_id)


    def delete_metadata(self, doc_id: str):
        if doc_id in self._metadatas:
            del self._metadatas[doc_id]


    def query_metadata_all(self) -> List[DocumentMetadata]:
        return list(self._metadatas.values())


    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        return [m for m in self._metadatas.values() if predicate(m.metadata)]


    # EMBEDDINGS
    def upsert_embedding(self, doc_id: str, embedding: List[float]):
        self._embeddings[doc_id] = embedding


    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        return self._embeddings.get(doc_id)


    def delete_embedding(self, doc_id: str):
        if doc_id in self._embeddings:
            del self._embeddings[doc_id]


    def all_ids_with_embeddings(self) -> List[str]:
        return list(self._embeddings.keys())