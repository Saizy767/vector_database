import logging
from etl.core.etl.base import BaseExtractor
from etl.core.connector.sql_connector import SQLConnector
from typing import Generator, List, Dict, Any, Optional
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)

class SQLExtractor(BaseExtractor):
    def __init__(self, connector: SQLConnector):
        self.connector = connector
        logger.debug("SQLExtractor initialized")
    
    def extract_all(self, table_name:str,
            columns: Optional[List[str]] = None
                    ) -> List[Dict[str, Any]]:
        columns_select = ", ".join(columns) if columns else "*"
        logger.info(f"Extracting all rows from {table_name} (columns: {columns_select})")
        with self.connector.connect() as session:
            query = text(f'SELECT {columns_select} FROM {table_name}')
            result = session.execute(query)
            rows = result.mappings().all()
            logger.debug(f"Extracted {len(rows)} rows")
            return [dict(row) for row in rows]
        
    def extract_batches(self, table_name:str,
                      batch_size: int = 100,
                      columns: Optional[List[str]] = None
                      ) -> Generator[List[Dict[str, Any]], None, None]:
        columns_select = ", ".join(columns) if columns else "*"
        offset = 0
        logger.info(f"Starting batch extraction from {table_name} (batch_size={batch_size})")
        while True:
            with self.connector.connect() as session:
                query = text(
                    f'SELECT {columns_select} FROM {table_name} LIMIT :limit OFFSET :offset'
                    )
                
                result = session.execute(query, {'limit': int(batch_size), 'offset': int(offset)})
                rows = result.mappings().all()
                if not rows:
                    break
                logger.debug(f"Batch at offset {offset}: {len(rows)} rows")
                yield [dict(row) for row in rows]
                offset += batch_size
        logger.info("Batch extraction completed")


