from sqlalchemy import Column, String, Text, JSON, Boolean, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class MetadataORM(Base):
    __tablename__ = "metadata"

    id = Column(String, primary_key=True)
    text = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    version = Column(Integer, default=1)
    is_deleted = Column(Boolean, default=False)
    similar_ids = Column(JSON, default=[])

    embedding = relationship("EmbeddingORM", back_populates="metadata", uselist=False)


class EmbeddingORM(Base):
    __tablename__ = "embeddings"

    id = Column(String, ForeignKey("metadata.id", ondelete="CASCADE"), primary_key=True)
    embedding = Column(JSON, nullable=False)  # universal JSON format
    is_deleted = Column(Boolean, default=False)

    metadata = relationship("MetadataORM", back_populates="embedding")