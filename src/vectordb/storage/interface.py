from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from vectordb.models import DocumentMetadata, DocumentEmbedding

class StorageInterface(ABC):
    # === Metadata methods ===
    @abstractmethod
    def upsert_metadata(self, meta: DocumentMetadata):
        pass

    @abstractmethod
    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        pass

    @abstractmethod
    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        pass

    # === Embedding methods ===
    @abstractmethod
    def upsert_embedding(self, emb: DocumentEmbedding):
        pass

    @abstractmethod
    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        pass

    @abstractmethod
    def all_ids_with_embeddings(self) -> List[str]:
        pass

    # === Management methods ===
    @abstractmethod
    def delete(self, doc_id: str, hard: bool = False):
        pass