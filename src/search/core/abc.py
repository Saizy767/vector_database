from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseSearchBackend(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int,
        min_similarity: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]: ...

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]: ...