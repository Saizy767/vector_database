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

from shared.embedding.sentence_transformer import SentenceTransformerEmbedding
from shared.embedding.bert import BERTEmbedder
from shared.models import EmbeddingChapter

logger = logging.getLogger(__name__)


class ETLPipeline:
    def __init__(self, settings: ETLSettings):
        self.settings = settings
        self._validate_settings()

    def _validate_settings(self):
        if not self.settings.embedding_columns:
            raise ValueError("embedding_columns must contain at least one column name in .env")
        if not self.settings.extract_table_name:
            raise ValueError("extract_table_name is required")
        if not self.settings.load_table_name:
            raise ValueError("load_table_name is required")
        if not self.settings.source_id:
            raise ValueError("source_id is required")

    def _create_embedding(self):
        if self.settings.embedding_provider == "sentence-transformers":
            return SentenceTransformerEmbedding(
                model_name=self.settings.embedding_model,
                device=self.settings.device,
            )
        elif self.settings.embedding_provider == "bert":
            return BERTEmbedder(
                model_name=self.settings.embedding_model,
                device=self.settings.device,
            )
        else:
            raise NotImplementedError(f"Provider '{self.settings.embedding_provider}' not supported yet")

    def _create_metadata_builder(self):
        field_mapping = {col: col for col in self.settings.metadata_columns} if self.settings.metadata_columns else {}
        return MetadataBuilder(field_mapping=field_mapping)

    async def run_async(self):
        logger.info("🔄 Start ETL in ASYNC")
        connector = AsyncSQLConnector(self.settings.db_url)
        schema_manager = AsyncSchemaManager(connector.engine)
        await schema_manager.initialize()

        embedding = self._create_embedding()
        splitter = SentenceSplitter()
        metadata_builder = self._create_metadata_builder()

        text_column = self.settings.embedding_columns[0]
        columns = self.settings.embedding_columns + (self.settings.metadata_columns or [])

        vdb = AsyncVectorDB(
            connector=connector,
            embedding=embedding,
            splitter=splitter,
            metadata_builder=metadata_builder,
            target_table=self.settings.load_table_name,
            batch_size=self.settings.batch_size,
            orm_class=EmbeddingChapter,
            metadata_columns=self.settings.metadata_columns,
        )

        logger.info(
            f"Begin async ETL: '{self.settings.extract_table_name}' → '{vdb.target_table}'"
        )

        async for batch in vdb.extractor.extract_batches(
            table_name=self.settings.extract_table_name,
            batch_size=vdb.batch_size,
            columns=columns,
        ):
            if not batch:
                continue
            transformed_chunks = await vdb.transformer.atransform(
                batch_rows=batch,
                text_column=text_column,
                source_id_column=self.settings.source_id,
            )
            if transformed_chunks:
                session = await connector.connect()
                async with session.begin():
                    await vdb.loader.load(session, transformed_chunks)

        logger.info("✅ Async ETL-pipeline complete.")

    def run_sync(self):
        logger.info("🔄 Start ETL in SYNC")
        connector = SQLConnector(self.settings.db_url)
        schema_manager = SchemaManager(connector.engine)
        schema_manager.initialize()

        embedding = self._create_embedding()
        splitter = SentenceSplitter()
        metadata_builder = self._create_metadata_builder()

        text_column = self.settings.embedding_columns[0]
        columns = self.settings.embedding_columns + (self.settings.metadata_columns or [])

        vdb = VectorDB(
            connector=connector,
            embedding=embedding,
            splitter=splitter,
            metadata_builder=metadata_builder,
            target_table=self.settings.load_table_name,
            batch_size=self.settings.batch_size,
            orm_class=EmbeddingChapter,
            metadata_columns=self.settings.metadata_columns,
        )

        logger.info(
            f"Begin sync ETL: '{self.settings.extract_table_name}' → '{vdb.target_table}'"
        )

        for batch in vdb.extractor.extract_batches(
            table_name=self.settings.extract_table_name,
            batch_size=vdb.batch_size,
            columns=columns,
        ):
            transformed_batches = vdb.transformer.transform(
                batch_rows=batch,
                text_column=text_column,
                source_id_column=self.settings.source_id,
            )
            for transformed_batch in transformed_batches:
                if transformed_batch:
                    vdb.loader.load(transformed_batch)

        logger.info("✅ Sync ETL-pipeline complete.")