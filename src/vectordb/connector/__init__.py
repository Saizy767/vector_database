# src/vectordb/connector/__init__.py
"""
Модуль коннекторов для баз данных.
"""
from .sql_connector import SQLConnector

__all__ = ["SQLConnector"]
