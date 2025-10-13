import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from sqlalchemy.sql import text
from vectordb.connector.async_sql_connector import AsyncSQLConnector

logger = logging.getLogger(__name__)


class AsyncSQLExtractor:
    def __init__(self, connector: AsyncSQLConnector):
        self.connector = connector
        logger.debug("AsyncSQLExtractor initialized")

    async def extract_all(
        self,
        table_name: str,
        columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        columns_select = ", ".join(columns) if columns else "*"
        logger.info(f"Async extracting all rows from {table_name} (columns: {columns_select})")
        
        async with self.connector.connect() as session:
            query = text(f'SELECT {columns_select} FROM {table_name}')
            result = await session.execute(query)
            rows = result.mappings().all()
            logger.debug(f"Extracted {len(rows)} rows")
            return [dict(row) for row in rows]

    async def extract_batches(
        self,
        table_name: str,
        batch_size: int = 100,
        columns: Optional[List[str]] = None
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        columns_select = ", ".join(columns) if columns else "*"
        offset = 0
        logger.info(f"Starting async batch extraction from {table_name} (batch_size={batch_size})")
        
        while True:
            session = await self.connector.connect()
            async with session.begin():
                query = text(
                    f'SELECT {columns_select} FROM {table_name} LIMIT :limit OFFSET :offset'
                )
                result = await session.execute(query, {'limit': batch_size, 'offset': offset})
                rows = result.mappings().all()
                if not rows:
                    break
                logger.debug(f"Batch at offset {offset}: {len(rows)} rows")
                yield [dict(row) for row in rows]
                offset += batch_size
        
        logger.info("Async batch extraction completed")