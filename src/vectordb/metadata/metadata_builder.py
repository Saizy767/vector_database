from .base import BaseMetadata

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, ValidationInfo
from uuid import uuid4

class MetadataModel(BaseModel):
    """
    Модель метаданных для одного чанка текста.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_index: int
    total_chunks: int
    source_id: Optional[str] = None
    data: Dict[str, Any]

    @model_validator(mode="after")
    def validate_chunk_index(self):
        if self.chunk_index >= self.total_chunks:
            raise ValueError(
                f"chunk_index ({self.chunk_index}) cannot be greater or equal to total_chunks ({self.total_chunks})"
            )
        if self.total_chunks <= 0:
            raise ValueError("total_chunks must be greater than 0")
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")
        return self
    

class MetadataBuilder(BaseMetadata):
    def __init__(
            self, 
            field_mapping: Optional[Dict[str, str]] = None,
            include_system_fields: bool = True
             ):
        self.field_mapping = field_mapping or {}
        self.include_system_fields = include_system_fields
    
    def _map_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {self.field_mapping.get(k, k): v for k, v in data.items()}

    def build(
        self,
        row_data: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        mapped = self._map_fields(row_data)

        metadata = MetadataModel(
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            source_id=source_id,
            data=mapped,
        )
        return metadata.model_dump()
