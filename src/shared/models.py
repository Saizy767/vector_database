from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from .types import UTF8JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

def create_embedding_model(dim: int, table_name: str = "embedding_chapter"):
    """
    Динамически создаёт ORM-модель с нужной размерностью эмбеддинга.
    """
    class EmbeddingChapter(Base):
        __tablename__ = table_name
        __table_args__ = {'schema': 'public'}

        id = Column(Integer, primary_key=True, autoincrement=True)
        chunk_text = Column(Text, nullable=False)
        embedding = Column(Vector(dim), nullable=False)
        metadata_ = Column("metadata", UTF8JSON, nullable=False)

        def __repr__(self):
            return f"<EmbeddingChapter(table={table_name}, dim={dim})>"

    return EmbeddingChapter