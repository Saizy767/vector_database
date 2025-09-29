from sqlalchemy import Column, String, Text, JSON, Boolean, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class MetadataORM(Base):
    __tablename__ = "metadata"

    id = Column(String, primary_key=True)
    text = Column(Text, nullable=True)
    meta_json = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    version = Column(Integer, default=1)
    is_deleted = Column(Boolean, default=False)
    similar_ids = Column(JSON, default=[])

    embedding = relationship("EmbeddingORM", back_populates="parent", uselist=False)


class EmbeddingORM(Base):
    __tablename__ = "embeddings"

    id = Column(String, ForeignKey("metadata.id", ondelete="CASCADE"), primary_key=True)
    embedding = Column(JSON, nullable=False)
    is_deleted = Column(Boolean, default=False)

    parent = relationship("MetadataORM", back_populates="embedding")