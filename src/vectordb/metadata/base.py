from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseMetadata(ABC):
    @abstractmethod
    def build(
        self,
        row_data: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return NotImplementedError
    