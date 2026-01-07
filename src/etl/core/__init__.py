"""
Ядро ETL-пайплайна (Extract, Transform, Load) для векторной базы данных VectorDB.

Этот пакет содержит фундаментальные компоненты для построения ETL-процессов:
- Базовые абстракции и интерфейсы для всех компонентов
- Реализации коннекторов к различным источникам данных
- Модули преобразования данных (чанкинг, эмбеддинги, метаданные)
- Синхронные и асинхронные исполнители (раннеры)
- Основные классы для работы с векторными базами данных

Структура пакета:
├── connector/     # Коннекторы к базам данных (SQL, async SQL)
├── etl/           # Базовые компоненты ETL-процесса
│   ├── base.py    # Абстрактные базовые классы
│   ├── extractors/ # Извлечение данных из источников
│   ├── transformers/ # Преобразование данных в векторы
│   └── loaders/   # Загрузка данных в хранилища
├── metadata/      # Генераторы метаданных для чанков
├── runner/        # Исполнители ETL-процессов (sync/async)
├── splitters/     # Алгоритмы разделения текста на чанки
├── vector_db.py   # Синхронная реализация VectorDB
└── async_vector_db.py # Асинхронная реализация VectorDB

Ключевые особенности:
- Модульная архитектура с четким разделением ответственности
- Поддержка синхронного и асинхронного режимов работы
- Расширяемость через интерфейсы и наследование
- Интеграция с различными бэкендами (PostgreSQL/pgvector, FAISS)
- Гибкая настройка через фабрики компонентов

Этот модуль является основой для всего ETL-процесса и обеспечивает:
1. Извлечение данных из различных источников
2. Преобразование текста в векторные представления
3. Загрузку обработанных данных в векторные хранилища

Пример использования:
from etl.core.vector_db import VectorDB
from etl.core.connector.sql_connector import SQLConnector
from etl.core.pipeline.extractors.sql_extractor import SQLExtractor
from etl.core.pipeline.transformers.transformer import Transformer
from etl.core.pipeline.loaders.sql_loader import SQLLoader

# Создание компонентов
connector = SQLConnector(db_url)
extractor = SQLExtractor(connector)
transformer = Transformer(embedder, splitter, metadata_builder)
loader = SQLLoader(connector, table_name, orm_class)

# Создание и запуск пайплайна
vdb = VectorDB(extractor, transformer, loader)
vdb.transform_table(source_table="books", text_column="content")

Архитектурные принципы:
- Инверсия зависимостей: зависимость от абстракций, а не от конкретных реализаций
- Единый интерфейс для синхронных и асинхронных операций
- Изоляция бизнес-логики от деталей реализации хранилищ
- Поддержка различных стратегий обработки данных
"""

from .vector_db import VectorDB
from .async_vector_db import AsyncVectorDB
from .connector.sql_connector import SQLConnector
from .connector.async_sql_connector import AsyncSQLConnector
from .splitters.sentence_splitter import SentenceSplitter
from .metadata.metadata_builder import MetadataBuilder
from .runner.sync_runner import SyncETLRunner
from .runner.async_runner import AsyncETLRunner

__all__ = [
    "VectorDB",
    "AsyncVectorDB",
    "SQLConnector",
    "AsyncSQLConnector",
    "SentenceSplitter",
    "MetadataBuilder",
    "SyncETLRunner",
    "AsyncETLRunner"
]