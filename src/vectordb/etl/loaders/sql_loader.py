from typing import List, Dict, Any, Optional, Type
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Table, MetaData, text

from vectordb.etl.base import BaseLoader
from vectordb.connector.sql_connector import SQLConnector

class SQLLoader(BaseLoader):
    def __init__(
            self,
            connector: SQLConnector,
            table_name:str,
            batch_size: int = 100,
            conflict_update: Optional[List[str]] = None,
            orm_class: Optional[Type] = None
            ):
        self.connector = connector
        self.table_name = table_name
        self.batch_size = batch_size
        self.conflict_update = conflict_update
        self.orm_class = orm_class
    
    def load(self, data: List[Dict[str, Any]]):
        if not data:
            return
        
        keys = data[0].keys()
        columns = ', '.join(keys)
        placeholders = ", ".join(f":{k}" for k in keys)

        if self.conflict_update:
            update_cols = ", ".join(f"{col}=EXCLUDED.{col}" for col in self.conflict_update)
            update_clause = f" ON CONFLICT ({', '.join(self.conflict_update)}) DO UPDATE SET {update_cols}"
            query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders}){update_clause}"
            
            try:
                with self.connector.connect() as session:
                    for i in range(0, len(data), self.batch_size):
                        batch = data[i:i+self.batch_size]
                        session.execute(text(query), batch)
                    session.commit()
            except SQLAlchemyError as e:
                print(f"Ошибка при загрузке данных: {e}")
                raise
        else:
            if not self.orm_class:
                raise ValueError("Для bulk_insert_mappings нужен orm_class")
            try:
                with self.connector.connect() as session:
                    for i in range(0, len(data), self.batch_size):
                        batch = data[i:i+self.batch_size]
                        session.bulk_insert_mappings(self.orm_class, batch)
                    session.commit()
            except SQLAlchemyError as e:
                print(f"Ошибка при загрузке данных: {e}")
                raise


    def _table_class(self, session):
        meta = MetaData(bind=session.bind)
        table = Table(self.table_name, meta, autoload_with=session.bind)
        return table