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
from .splitters.sentence_splitter import SentenceSpliter

__all__ = [
    "VectorDB",
    "SQLConnector",
    "SentenceTransformerEmbedding",
    "MetadataBuilder",
    "SentenceSpliter",
    "BERTEmbedder"
]
