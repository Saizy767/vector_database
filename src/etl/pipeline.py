import logging

from etl.config import ETLSettings
from etl.core.vector_db import VectorDB
from etl.core.async_vector_db import AsyncVectorDB
from etl.core.connector.sql_connector import SQLConnector
from etl.core.connector.async_sql_connector import AsyncSQLConnector
from etl.core.splitters.sentence_splitter import SentenceSplitter
from etl.core.metadata.metadata_builder import MetadataBuilder
from etl.schema.schema_manager import SchemaManager
from etl.schema.async_schema_manager import AsyncSchemaManager
from etl.core.runner.abc import IETLRunner
from etl.core.runner.sync_runner import SyncETLRunner
from etl.core.runner.async_runner import AsyncETLRunner

from shared.embedding.sentence_transformer import SentenceTransformerEmbedding
from shared.embedding.bert import BERTEmbedder
from shared.models import create_embedding_model

logger = logging.getLogger(__name__)


class ETLPipeline:
    def __init__(self, settings: ETLSettings):
        self.settings = settings
        self._validate_settings()
        self.runner: IETLRunner = self._create_runner()

    def _validate_settings(self):
        if not self.settings.embedding_columns:
            raise ValueError("embedding_columns must contain at least one column name in .env")
        if not self.settings.extract_table_name:
            raise ValueError("extract_table_name is required")
        if not self.settings.load_table_name:
            raise ValueError("load_table_name is required")
        if not self.settings.source_id:
            raise ValueError("source_id is required")

    def _create_runner(self) -> IETLRunner:
        if self.settings.async_mode:
            logger.info("Создание AsyncETLRunner")
            return AsyncETLRunner(self.settings)
        else:
            logger.info("Создание SyncETLRunner")
            return SyncETLRunner(self.settings)
    
    async def run(self) -> None:
        """Единая точка входа — всегда async."""
        logger.info("Запуск ETL через runner")
        await self.runner.run()
