"""
Фабрики компонентов для ETL-пайплайна VectorDB.

Этот пакет содержит фабрики для создания компонентов ETL-процесса
в различных режимах выполнения (синхронном и асинхронном).

Структура пакета:
├── base_factory.py     # Абстрактная базовая фабрика с общим интерфейсом
├── sync_factory.py     # Фабрика для синхронного режима работы
└── async_factory.py    # Фабрика для асинхронного режима работы

Основные возможности:
- Централизованное создание компонентов ETL-пайплайна
- Поддержка различных режимов выполнения (sync/async)
- Инкапсуляция логики инициализации зависимостей
- Гибкость в настройке компонентов через конфигурацию

Ключевые компоненты, создаваемые фабриками:
- Коннекторы к базам данных (SQLConnector, AsyncSQLConnector)
- Генераторы эмбеддингов (SentenceTransformerEmbedding, BERTEmbedder)
- Разделители текста (SentenceSplitter, SemanticChunker)
- Генераторы метаданных (MetadataBuilder)
- Экстракторы данных (SQLExtractor, AsyncSQLExtractor)
- Трансформеры данных (Transformer)
- Загрузчики данных (SQLLoader, AsyncSQLLoader, FAISSLoader)
- ORM-модели для векторных баз данных

Принцип работы:
1. Фабрика принимает конфигурацию (ETLSettings) в конструкторе
2. На основе режима работы (async_mode) выбирается соответствующая фабрика
3. Фабрика создает все необходимые компоненты с правильными зависимостями
4. Компоненты интегрируются в единый ETL-пайплайн через VectorDB

Пример использования:
from etl.config import ETLSettings
from etl.factory import get_factory

settings = ETLSettings()
factory = get_factory(settings)

# Создание компонентов
connector = factory.create_connector()
embedder = factory.create_embedder()
splitter = factory.create_splitter()
metadata_builder = factory.create_metadata_builder()
extractor = factory.create_extractor(connector)
orm_model = factory.create_orm_model(factory.get_embedding_dim(embedder))
loader = factory.create_loader(connector=connector, orm_class=orm_model)
transformer = factory.create_transformer(embedder, splitter, metadata_builder)

# Интеграция в пайплайн
vdb = VectorDB(extractor, transformer, loader)
Выбор фабрики определяется параметром async_mode в конфигурации:

async_mode = False → SyncComponentFactory
async_mode = True → AsyncComponentFactory
Этот пакет реализует паттерн "Абстрактная фабрика" для обеспечения гибкости
и расширяемости ETL-архитектуры.
"""

from .base_factory import BaseComponentFactory
from .sync_factory import SyncComponentFactory
from .async_factory import AsyncComponentFactory


def get_factory(settings):
    """
        Фабричный метод для получения подходящей фабрики компонентов
        на основе конфигурации.
        Args:
            settings (ETLSettings): Конфигурация ETL-процесса
            
        Returns:
            BaseComponentFactory: Экземпляр фабрики (SyncComponentFactory или AsyncComponentFactory)
            
        Raises:
            ValueError: Если указан неподдерживаемый режим работы
    """
    if settings.async_mode:
        return AsyncComponentFactory(settings)
    return SyncComponentFactory(settings)


__all__ = [
    "BaseComponentFactory",
    "SyncComponentFactory",
    "AsyncComponentFactory",
    "get_factory"
]




all = [
    "BaseComponentFactory",
    "SyncComponentFactory",
    "AsyncComponentFactory",
    "get_factory"
]