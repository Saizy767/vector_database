import logging
from typing import Optional, List
from vectordb.connector.async_sql_connector import AsyncSQLConnector
from vectordb.etl.extractors.async_sql_extractor import AsyncSQLExtractor
from vectordb.etl.transformers.transformer import Transformer
from vectordb.etl.loaders.async_sql_loader import AsyncSQLLoader
from vectordb.embedding.base import BaseEmbedding
from vectordb.splitters.base import BaseSplitter
from vectordb.metadata.metadata_builder import MetadataBuilder

logger = logging.getLogger(__name__)


class AsyncVectorDB:
    def __init__(
        self,
        connector: AsyncSQLConnector,
        embedding: BaseEmbedding,
        splitter: BaseSplitter,
        metadata_builder: Optional[MetadataBuilder] = None,
        target_table: Optional[str] = None,
        batch_size: int = 100,
        orm_class=None,
        metadata_columns: Optional[List[str]] = None,
    ):
        self.connector = connector
        self.batch_size = batch_size
        self.target_table = target_table
        self.orm_class = orm_class

        self.extractor = AsyncSQLExtractor(connector)
        self.transformer = Transformer(
            embedding=embedding,
            splitter=splitter,
            metadata_builder=metadata_builder,
            metadata_columns=metadata_columns,
        )
        self.loader = AsyncSQLLoader(
            orm_class=orm_class,
            batch_size=batch_size,
        )
        logger.info("Intialized AsyncVectorDB with async extractor, transformer, loader")

    async def async_transform_table(
        self,
        source_table: str,
        text_column: str,
        source_id_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        if not text_column:
            raise ValueError("text_column is required")
        if not self.target_table:
            raise ValueError("target_table must be specified")
        if self.orm_class is None:
            raise ValueError("orm_class is required for loading into ORM table")

        logger.info(f"Starting async transform from '{source_table}' to '{self.target_table}'")

        async for batch in self.extractor.extract_batches(
            table_name=source_table,
            batch_size=self.batch_size,
            columns=columns,
        ):
            if not batch:
                continue

            transformed_chunks = await self.transformer.atransform(
                batch_rows=batch,
                text_column=text_column,
                source_id_column=source_id_column,
            )

            if transformed_chunks:
                async with self.connector.connect() as session:
                    await self.loader.load(session, transformed_chunks)

        logger.info("✅ Async ETL pipeline completed successfully.")