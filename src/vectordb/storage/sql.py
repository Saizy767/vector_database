from typing import List, Optional, Callable
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from vectordb.storage.interface import StorageInterface
from vectordb.models import DocumentMetadata, DocumentEmbedding
from vectordb.storage.sql_base import Base, MetadataORM, EmbeddingORM


class SQLStorage(StorageInterface):
    """
    SQLAlchemy-based storage backend for metadata and embeddings.
    connection_string: SQLAlchemy connect string, e.g.:
    sqlite:///vectordb.sqlite
    postgresql+psycopg2://user:pass@host/dbname
    """

    def __init__(self, connection_string: str, echo: bool = False):
        self.engine = create_engine(connection_string, echo=echo, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # ----------------- Metadata -----------------

    def upsert_metadata(self, meta: DocumentMetadata):
        with self.Session() as session:
            orm_obj = session.get(MetadataORM, meta.id)
            if orm_obj is None:
                orm_obj = MetadataORM(
                    id=meta.id,
                    text=meta.text,
                    meta_json=meta.metadata or {},  # ✅ исправлено
                    version=meta.version,
                    is_deleted=meta.is_deleted,
                    similar_ids=meta.similar_ids or [],
                )
                session.add(orm_obj)
            else:
                orm_obj.text = meta.text
                orm_obj.meta_json = meta.metadata or {}  # ✅ исправлено
                orm_obj.version = meta.version
                orm_obj.is_deleted = meta.is_deleted
                orm_obj.similar_ids = meta.similar_ids or []
            session.commit()

    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        with self.Session() as session:
            orm_obj = session.get(MetadataORM, doc_id)
            if orm_obj is None:
                return None
            return DocumentMetadata(
                id=orm_obj.id,
                metadata=orm_obj.meta_json or {},
                text=orm_obj.text,
                created_at=orm_obj.created_at.timestamp() if orm_obj.created_at else None,
                updated_at=orm_obj.updated_at.timestamp() if orm_obj.updated_at else None,
                version=orm_obj.version,
                is_deleted=orm_obj.is_deleted,
                similar_ids=orm_obj.similar_ids or [],
            )

    def query_metadata(self, predicate: Callable[[dict], bool]) -> List[DocumentMetadata]:
        results: List[DocumentMetadata] = []
        with self.Session() as session:
            rows = session.query(MetadataORM).all()
            for r in rows:
                if predicate(r.meta_json):
                    results.append(
                        DocumentMetadata(
                            id=r.id,
                            metadata=r.meta_json or {},
                            text=r.text,
                            created_at=r.created_at.timestamp() if r.created_at else None,
                            updated_at=r.updated_at.timestamp() if r.updated_at else None,
                            version=r.version,
                            is_deleted=r.is_deleted,
                            similar_ids=r.similar_ids or []
                        )
                    )
        return results

    # ----------------- Embeddings -----------------

    def upsert_embedding(self, emb: DocumentEmbedding):
        with self.Session() as session:
            orm_obj = session.get(EmbeddingORM, emb.id)
            if orm_obj is None:
                orm_obj = EmbeddingORM(
                    id=emb.id,
                    embedding=emb.embedding,
                    is_deleted=emb.is_deleted,
                )
                session.add(orm_obj)
            else:
                orm_obj.embedding = emb.embedding
                orm_obj.is_deleted = emb.is_deleted
            session.commit()

    def get_embedding(self, doc_id: str) -> Optional[List[float]]:  # ✅ исправлено
        with self.Session() as session:
            orm_obj = session.get(EmbeddingORM, doc_id)
            if orm_obj is None:
                return None
            return orm_obj.embedding

    def all_ids_with_embeddings(self) -> List[str]:
        with self.Session() as session:
            rows = session.query(EmbeddingORM.id).all()
            return [r[0] for r in rows]

    # ----------------- Delete -----------------

    def delete(self, doc_id: str, hard: bool = False):
        with self.Session() as session:
            meta = session.get(MetadataORM, doc_id)
            emb = session.get(EmbeddingORM, doc_id)
            if hard:
                if emb:
                    session.delete(emb)
                if meta:
                    session.delete(meta)
            else:
                if meta:
                    meta.is_deleted = True
                if emb:
                    emb.is_deleted = True
            session.commit()
