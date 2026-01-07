"""
Экстракторы данных из различных источников для VectorDB ETL пайплайна.

Содержит реализации извлечения данных из разных типов источников:
- sql_extractor.py: Извлечение данных из PostgreSQL базы данных
- async_sql_extractor.py: Асинхронное извлечение из PostgreSQL
- faiss_extractor.py: Извлечение данных из FAISS индекса

Общие возможности всех экстракторов:
- Поддержка пакетного извлечения данных
- Фильтрация колонок для выборки
- Обработка больших объемов данных через генераторы
- Логирование процесса извлечения

Интерфейс экстракторов:
- extract_all(): Извлечение всех данных сразу
- extract_batches(): Генератор для пакетного извлечения

Пример использования:
from etl.core.pipeline.extractors import SQLExtractor
from etl.core.connector import SQLConnector

connector = SQLConnector("postgresql://...")
extractor = SQLExtractor(connector)

# Извлечение всех данных
all_data = extractor.extract_all("books_table")

# Пакетное извлечение
for batch in extractor.extract_batches("books_table", batch_size=100):
    process_batch(batch)

Этот модуль обеспечивает гибкость в выборе источников данных и поддерживает
различные режимы работы (синхронный/асинхронный, пакетный/полный).
"""

from .sql_extractor import SQLExtractor

__all__ = ["SQLExtractor"]
