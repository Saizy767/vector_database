from __future__ import annotations
from typing import Dict, List, Optional
from src.vectordb.models import DocumentMetadata

class MemoryAdapter:
    """Storage adapter that keeps documents in memory.
    Acts as the persistence layer; index layers can be separate.
    """
    def __init__(self):
        self._docs: Dict[str, DocumentMetadata] = {}

    def upsert(self, doc: DocumentMetadata):
        self._docs[doc.id] = doc

    def get(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self._docs.get(doc_id)

    def delete(self, doc_id: str):
        if doc_id in self._docs:
            del self._docs[doc_id]

    def query_all(self) -> List[DocumentMetadata]:
        return list(self._docs.values())

    def query_by_metadata(self, predicate) -> List[DocumentMetadata]:
        return [d for d in self._docs.values() if predicate(d.metadata)]
