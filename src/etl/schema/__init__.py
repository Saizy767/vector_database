"""
Модуль управления схемой базы данных для VectorDB.

Предоставляет инструменты для инициализации и управления схемой PostgreSQL базы данных:
- schema_manager.py: Менеджер для синхронной работы с БД
- async_schema_manager.py: Менеджер для асинхронной работы с БД

Ключевые возможности:
- Автоматическое создание расширения pgvector
- Создание и обновление таблиц по ORM моделям
- Проверка существования схемы перед операциями
- Поддержка миграций (в разработке)

Использование:

from etl.schema import SchemaManager
from etl.core.connector import SQLConnector

connector = SQLConnector("postgresql://...")
schema_manager = SchemaManager(connector.engine)
schema_manager.initialize()  # Создает расширение и таблицы

# Асинхронная версия
from etl.schema import AsyncSchemaManager
schema_manager = AsyncSchemaManager(connector.engine)
await schema_manager.initialize()

Этот модуль гарантирует, что база данных всегда имеет правильную структуру
перед началом ETL процесса, что предотвращает ошибки выполнения.
"""

from .schema_manager import SchemaManager
__all__ = ["SchemaManager"]