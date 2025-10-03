from typing import List, Dict, Any, Optional, Type
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Table, MetaData, insert
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.mysql import insert as mysql_insert

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
                    self.upsert(session, data)
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
        meta = MetaData()
        table = Table(self.table_name, meta, autoload_with=session.bind)
        return table
    

    def upsert(self, session: Session, data: list[dict], pk: str = "id"):
        if not data:
            return
        
        table = self._table_class(session)
        dialect = session.bind.dialect.name

        if dialect == "postgresql":
            stmt = pg_insert(table).values(data)
            update_dict = {c.name: stmt.excluded[c.name] for c in table.columns if c.name != pk}
            stmt = stmt.on_conflict_do_update(index_elements=[pk], set_=update_dict)

        elif dialect in ("mysql", "mariadb"):
            stmt = mysql_insert(table).values(data)
            update_dict = {c.name: stmt.inserted[c.name] for c in table.columns if c.name != pk}
            stmt = stmt.on_duplicate_key_update(**update_dict)

        elif dialect == "sqlite":
            stmt = insert(table).values(data)
            update_dict = {c.name: stmt.excluded[c.name] for c in table.columns if c.name != pk}
            stmt = stmt.on_conflict_do_update(index_elements=[pk], set_=update_dict)

        else:
            raise NotImplementedError(f"Upsert не поддержан для {dialect}")

        session.execute(stmt)