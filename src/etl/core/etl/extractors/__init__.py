# src/vectordb/etl/extractors/__init__.py
"""
Экстракторы данных из различных источников.
"""
from .sql_extractor import SQLExtractor

__all__ = ["SQLExtractor"]
