from __future__ import annotations
from typing import List, Optional, Callable


try:
    import sqlalchemy as sa
    from sqlalchemy import Table, Column, String, MetaData, JSON
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


from vectordb.storage.interface import StorageInterface
from vectordb.models import DocumentMetadata


class SQLStorage(StorageInterface):
    def __init__(self, connection_string: str):
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not installed; install it to use SQLStorage")
        self.engine = sa.create_engine(connection_string)
        self.meta = MetaData()
        self.meta_table = Table(
        'vectordb_metadata', self.meta,
        Column('id', String, primary_key=True),
        Column('text', String),
        Column('metadata', JSON),
        )
        self.emb_table = Table(
        'vectordb_embeddings', self.meta,
        Column('id', String, primary_key=True),
        Column('embedding', String),
        )
        self.meta.create_all(self.engine)


    def upsert_metadata(self, meta: DocumentMetadata):
        with self.engine.begin() as conn:
            stmt = sa.insert(self.meta_table).values(id=meta.id, text=meta.text, metadata=meta.metadata)
            stmt = stmt.prefix_with('OR REPLACE')
            conn.execute(stmt)


    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        with self.engine.connect() as conn:
            q = self.meta_table.select().where(self.meta_table.c.id == doc_id)
            r = conn.execute(q).first()
            if r is None:
                return None
            return DocumentMetadata(id=r['id'], text=r.get('text'), metadata=r.get('metadata'))


    def delete_metadata(self, doc_id: str):
        with self.engine.begin() as conn:
            stmt = self.meta_table.delete().where(self.meta_table.c.id == doc_id)
            conn.execute(stmt)


    def query_metadata_all(self) -> List[DocumentMetadata]:
        with self.engine.connect() as conn:
            res = conn.execute(self.meta_table.select()).fetchall()
            return [DocumentMetadata(id=r['id'], text=r.get('text'), metadata=r.get('metadata')) for r in res]


    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        return [m for m in self.query_metadata_all() if predicate(m.metadata)]


    def upsert_embedding(self, doc_id: str, embedding: List[float]):
        with self.engine.begin() as conn:
            stmt = sa.insert(self.emb_table).values(id=doc_id, embedding=str(embedding))
            stmt = stmt.prefix_with('OR REPLACE')
            conn.execute(stmt)


    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        with self.engine.connect() as conn:
            q = self.emb_table.select().where(self.emb_table.c.id == doc_id)
            r = conn.execute(q).first()
            if r is None:
                return None
            return eval(r['embedding'])
        
    
    def delete_embedding(self, doc_id: str):
        with self.engine.begin() as conn:
            stmt = self.emb_table.delete().where(self.emb_table.c.id == doc_id)
            conn.execute(stmt)


    def all_ids_with_embeddings(self) -> List[str]:
        with self.engine.connect() as conn:
            res = conn.execute(self.emb_table.select()).fetchall()
            return [r['id'] for r in res]