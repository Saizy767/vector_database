"""
Загрузчики данных в различные источники для VectorDB ETL пайплайна.

Содержит реализации загрузки обработанных данных в разные типы хранилищ:
- sql_loader.py: Загрузка данных в PostgreSQL/pgvector
- async_sql_loader.py: Асинхронная загрузка в PostgreSQL/pgvector
- faiss_loader.py: Загрузка данных в локальный FAISS индекс

Общие возможности всех загрузчиков:
- Поддержка пакетной загрузки
- Обработка конфликтов (upsert)
- Валидация данных перед загрузкой
- Откат транзакций при ошибках

Типы загрузки:
- bulk_insert: Массовая вставка новых данных
- upsert: Обновление существующих или вставка новых записей

Пример использования:

from etl.core.pipeline.loaders import SQLLoader
from etl.core.connector import SQLConnector

connector = SQLConnector("postgresql://...")
loader = SQLLoader(
    connector=connector,
    table_name="embeddings",
    orm_class=EmbeddingModel
)

# Загрузка данных
loader.load(transformed_data)

# Загрузка с обновлением
loader = SQLLoader(
    connector=connector,
    table_name="embeddings",
    conflict_update=["embedding", "metadata_"],
    conflict_target=["chunk_id"]
)
loader.load(transformed_data)

Этот модуль обеспечивает надежность и производительность при загрузке данных,
поддерживая различные стратегии в зависимости от требований к данным и хранилищу.
"""

from .sql_loader import SQLLoader

__all__ = ["SQLLoader"]
