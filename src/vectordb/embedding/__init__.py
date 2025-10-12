# src/vectordb/embedding/__init__.py
"""
Модуль для создания эмбеддингов текста.
"""
from .sentence_transformer import SentenceTransformerEmbedding
from .bert import BERTEmbedder

__all__ = [
    "SentenceTransformerEmbedding",
    "BERTEmbedder"
    ]
