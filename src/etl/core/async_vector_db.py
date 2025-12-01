import logging
from typing import Optional, List
from etl.core.connector.async_sql_connector import AsyncSQLConnector
from etl.core.etl.extractors.async_sql_extractor import AsyncSQLExtractor
from etl.core.etl.transformers.transformer import Transformer
from etl.core.etl.loaders.async_sql_loader import AsyncSQLLoader
from etl.core.etl.base import BaseExtractor, BaseLoader, BaseTransformer

logger = logging.getLogger(__name__)


class AsyncVectorDB:
    def __init__(
        self,
        extractor: BaseExtractor,
        transformer: BaseTransformer,
        loader: BaseLoader,
        batch_size: int = 100,
    ):
        """
        Универсальный асинхронный VectorDB.
        Поддерживает только SQL-based loader'ы (FAISS не поддерживает async).
        """
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.batch_size = batch_size
        logger.info("Initialized AsyncVectorDB with extractor, transformer, loader")

    async def async_transform_table(
        self,
        source_table: str,
        text_column: str,
        source_id_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        if not text_column:
            raise ValueError("text_column is required")
        logger.info(f"Starting async transform from '{source_table}'")
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
                if hasattr(self.loader, "orm_class"):
                    session = await self.extractor.connector.connect()
                    try:
                        await self.loader.load(session, transformed_chunks)
                        await session.commit()
                    except Exception:
                        await session.rollback()
                        raise
                    finally:
                        await session.close()
                else:
                    await self.loader.load(transformed_chunks)

        logger.info("✅ Async ETL pipeline completed successfully.")