"""
Модуль трансформеров для ETL-пайплайна VectorDB.

Этот пакет содержит компоненты для преобразования извлеченных данных
в векторные представления и структурированные чанки для загрузки в векторную базу данных.

Основные задачи трансформеров:
- Разбиение текста на семантически осмысленные чанки
- Генерация векторных эмбеддингов для каждого чанка
- Создание и управление метаданными для чанков
- Обработка ошибок и валидация данных

Структура пакета:
├── transformer.py    # Основной класс Transformer для синхронного и асинхронного преобразования
└── __init__.py       # Точка входа и экспорт компонентов

Ключевые возможности:
- Поддержка как синхронного, так и асинхронного преобразования данных
- Интеграция с различными эмбеддерами (Sentence Transformers, BERT)
- Гибкая настройка алгоритмов чанкинга через интерфейс BaseSplitter
- Расширяемая система метаданных через MetadataBuilder
- Обработка ошибок и пропуск некорректных данных с логированием

Пример использования:
from etl.core.pipeline.transformers import Transformer
from shared.embedding.sentence_transformer import SentenceTransformerEmbedding
from etl.core.splitters.sentence_splitter import SentenceSplitter
from etl.core.metadata.metadata_builder import MetadataBuilder

# Создание компонентов
embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
splitter = SentenceSplitter()
metadata_builder = MetadataBuilder(field_mapping={'author': 'doc_author'})

# Инициализация трансформера
transformer = Transformer(
    embedding=embedder,
    splitter=splitter,
    metadata_builder=metadata_builder,
    metadata_columns=['author', 'title', 'year']
)

# Синхронное преобразование
transformed_data = transformer.transform(
    batch_rows=extracted_data,
    text_column='content',
    source_id_column='book_id'
)

# Асинхронное преобразование
transformed_data = await transformer.atransform(
    batch_rows=extracted_data,
    text_column='content',
    source_id_column='book_id'
)
Архитектурные принципы:
- Разделение ответственности между чанкингом, эмбеддингами и метаданными
- Поддержка расширяемости через интерфейсы
- Обработка ошибок на уровне отдельных чанков без прерывания всего пайплайна
- Оптимизация производительности через пакетную обработку
- Сохранение контекста исходных данных в метаданных
- Этот модуль является критически важным компонентом ETL-пайплайна, так как определяет
качество векторных представлений и структуру данных для последующего семантического поиска.
"""

from .transformer import Transformer

__all__ = [
    "Transformer"
]