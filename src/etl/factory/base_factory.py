from abc import ABC, abstractmethod
from etl.config import ETLSettings
from etl.core.connector.base import BaseConnector
from shared.embedding.base import BaseEmbedding
from etl.core.splitters.base import BaseSplitter
from etl.core.metadata.base import BaseMetadata
from etl.core.etl.base import BaseExtractor, BaseLoader, BaseTransformer

class BaseComponentFactory(ABC):
    def __init__(self, settings: ETLSettings):
        self.settings = settings

    @abstractmethod
    def create_connector(self) -> BaseConnector:
        pass

    @abstractmethod
    def create_embedder(self) -> BaseEmbedding:
        pass

    @abstractmethod
    def get_embedding_dim(self, embedder: BaseEmbedding) -> int:
        pass

    @abstractmethod
    def create_orm_model(self, embedding_dim: int):
        pass

    @abstractmethod
    def create_splitter(self) -> BaseSplitter:
        pass

    @abstractmethod
    def create_metadata_builder(self) -> BaseMetadata:
        pass

    @abstractmethod
    def create_extractor(self, connector) -> BaseExtractor:
        pass

    @abstractmethod
    def create_loader(self, **kwargs) -> BaseLoader:
        pass

    @abstractmethod
    def create_transformer(self, 
                           embedder: BaseEmbedding,
                           splitter: BaseSplitter,
                           metadata_builder: BaseMetadata,
                           **extra_options,
                           ) -> BaseTransformer:
        pass

    def get_columns(self) -> list[str]:
        """Общая логика — может быть реализована здесь."""
        cols = self.settings.embedding_columns.copy()
        if self.settings.metadata_columns:
            cols.extend(self.settings.metadata_columns)
        return cols

    def get_text_column(self) -> str:
        if not self.settings.embedding_columns:
            raise ValueError("embedding_columns must be non-empty")
        return self.settings.embedding_columns[0]
    
    