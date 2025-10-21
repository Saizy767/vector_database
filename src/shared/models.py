from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from .types import UTF8JSON
from etl.config import settings
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class EmbeddingChapter(Base):
    __tablename__ = settings.load_table_name
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    metadata_ = Column("metadata", UTF8JSON, nullable=False)