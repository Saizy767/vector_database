# src/vectordb/__init__.py
"""
VectorDB — модуль для работы с векторными базами данных.

Включает:
- коннекторы (SQLConnector),
- загрузчики/экстракторы,
- создание эмбеддингов,
- работу с метаданными и чанками текста.
"""

from .vector_db import VectorDB
from .connector.sql_connector import SQLConnector
from .embedding.sentence_transformer import SentenceTransformerEmbedding
from .embedding.bert import BERTEmbedder
from .metadata.metadata_builder import MetadataBuilder
from .splitters.sentence_splitter import SentenceSplitter
from .schema_manager.schema_manager import SchemaManager

__all__ = [
    "VectorDB",
    "SQLConnector",
    "SentenceTransformerEmbedding",
    "MetadataBuilder",
    "SentenceSplitter",
    "BERTEmbedder",
    "SchemaManager"
]
