from vectordb.etl.base import BaseExtractor
from vectordb.connector.sql_connector import SQLConnector
from typing import Generator, List, Dict, Any, Optional
from sqlalchemy.sql import text

class SQLExtractor(BaseExtractor):
    def __init__(self, connector: SQLConnector):
        self.connector = connector
    
    def extract_all(self, table_name:str,
            columns: Optional[List[str]] = None
                    ) -> List[Dict[str, Any]]:
        columns_select = ", ".join(columns) if columns else "*"
        with self.connector as session:
            query = text(f'SELECT {columns_select} FROM {table_name}')
            result = session.execute(query)
            rows = result.mappings().all()
            return [dict(row) for row in rows]
        
    def extract_batch(self, table_name:str,
                      batch_size: int = 100,
                      columns: Optional[List[str]] = None
                      ) -> Generator[List[Dict[str, Any]], None, None]:
        columns_select = ", ".join(columns) if columns else "*"
        offset = 0
        while True:
            with self.connector as session:
                query = text(
                    f'SELECT {columns_select} FROM {table_name} LIMIT :limit OFFSET :offset'
                    )
                result = session.execute(query, {'limit': batch_size, 'offset': offset})
                rows = result.mappings().all()
                if not rows:
                    break
                yield [dict(row) for row in rows]

                offset += batch_size


