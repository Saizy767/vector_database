# src/vectordb/etl/loaders/__init__.py
"""
Загрузчики данных в различные источники (SQL, векторные БД и т.д.).
"""
from .sql_loader import SQLLoader

__all__ = ["SQLLoader"]
