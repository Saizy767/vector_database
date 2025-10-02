from typing import List, Dict, Any, Optional, Type
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Table, MetaData, text
from sqlalchemy.orm import Session

from vectordb.etl.base import BaseLoader
from vectordb.connector.sql_connector import SQLConnector

class SQLLoader(BaseLoader):
    def __init__(
            self,
            connector: SQLConnector,
            table_name:str,
            batch_size: int = 100,
            conflict_update: Optional[List[str]] = None,
            conflict_target: Optional[List[str]] = None,
            orm_class: Optional[Type] = None
            ):
        self.connector = connector
        self.table_name = table_name
        self.batch_size = batch_size
        self.conflict_update = conflict_update
        self.conflict_target = conflict_target
        self.orm_class = orm_class
    
    def load(self, data: List[Dict[str, Any]]):
        if not data:
            return
        
        try:
            with self.connector.connect() as session:  # 🔑 SQLConnector.connect()
                if self.conflict_update:
                    self._upsert(session, data)
                else:
                    self._bulk_insert(session, data)
                session.commit()
        except SQLAlchemyError as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise


    def _bulk_insert(self, session: Session, data: List[Dict]):
        if self.orm_class is None:
            raise ValueError("Для bulk insert требуется orm_class")

        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            session.bulk_insert_mappings(self.orm_class, batch)


    def _table_class(self, session: Session):
        meta = MetaData(bind=session.bind)
        table = Table(self.table_name, meta, autoload_with=session.bind)
        return table
    

    def _upsert(self, session: Session, data: List[Dict]):
        if not self.conflict_target:
            raise ValueError(
                "Для UPSERT нужно указать conflict_target (PK или UNIQUE колонки)"
            )

        columns = list(data[0].keys())
        insert_cols = ", ".join(columns)
        insert_vals = ", ".join([f":{col}" for col in columns])

        update_cols = ", ".join(
            [f"{col}=EXCLUDED.{col}" for col in self.conflict_update]
        )
        conflict_cols = ", ".join(self.conflict_target)

        query = text(
            f"""
            INSERT INTO {self.table_name} ({insert_cols})
            VALUES ({insert_vals})
            ON CONFLICT ({conflict_cols})
            DO UPDATE SET {update_cols}
            """
        )

        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            session.execute(query, batch)