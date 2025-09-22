from typing import List, Optional, Callable, Dict
from vectordb.storage.interface import StorageInterface
from vectordb.models import DocumentMetadata, DocumentEmbedding


class StubStorage(StorageInterface):
    """Stub storage with in-memory storage to fully test VectorDB"""
    def __init__(self):
        self._metadatas: Dict[str, DocumentMetadata] = {}
        self._embeddings: Dict[str, DocumentEmbedding] = {}


    def upsert_metadata(self, meta: DocumentMetadata):
        print(f"Stub upsert_metadata called: {meta.id}")
        self._metadatas[meta.id] = meta


    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        print(f"Stub get_metadata called: {doc_id}")
        return self._metadatas.get(doc_id)


    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        print("Stub query_metadata called")
        return [m for m in self._metadatas.values() if predicate(m.metadata)]


    def upsert_embedding(self, emb: DocumentEmbedding):
        print(f"Stub upsert_embedding called: {emb.id}")
        self._embeddings[emb.id] = emb


    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        print(f"Stub get_embedding called: {doc_id}")
        emb_obj = self._embeddings.get(doc_id)
        return emb_obj.embedding if emb_obj else None


    def all_ids_with_embeddings(self) -> List[str]:
        print("Stub all_ids_with_embeddings called")
        return list(self._embeddings.keys())


    def delete(self, doc_id: str, hard: bool = False):
        print(f"Stub delete called: {doc_id}, hard={hard}")
        if doc_id in self._metadatas:
            self._metadatas[doc_id].is_deleted = True
        if doc_id in self._embeddings:
            self._embeddings[doc_id].is_deleted = True