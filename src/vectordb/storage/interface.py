from __future__ import annotations
from typing import List, Optional, Callable
from vectordb.models import DocumentMetadata

class StorageInterface:
    """Interface for storage that separates embeddings and metadata.
    Concrete implementations must provide these methods.
    """
    # METADATA methods
    def upsert_metadata(self, meta: DocumentMetadata):
        raise NotImplementedError

    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        raise NotImplementedError

    def delete_metadata(self, doc_id: str):
        raise NotImplementedError

    def query_metadata_all(self) -> List[DocumentMetadata]:
        raise NotImplementedError

    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        raise NotImplementedError

    # EMBEDDING methods
    def upsert_embedding(self, doc_id: str, embedding: List[float]):
        raise NotImplementedError

    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        raise NotImplementedError

    def delete_embedding(self, doc_id: str):
        raise NotImplementedError

    def all_ids_with_embeddings(self) -> List[str]:
        raise NotImplementedError

    # Combined utility
    def delete(self, doc_id: str):
        self.delete_embedding(doc_id)
        self.delete_metadata(doc_id)