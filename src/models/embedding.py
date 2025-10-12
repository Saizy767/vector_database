from sqlalchemy import Column, Integer, Text, JSON
from .base import Base

class EmbeddingChapter(Base):
    __tablename__ = 'embedding_chapter'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=False)