"""
ETL (Extract, Transform, Load) пайплайн для векторной базы данных.

Этот пакет отвечает за:
- Извлечение данных из различных источников (PostgreSQL, FAISS)
- Преобразование текста в векторные представления (эмбеддинги)
- Загрузку обработанных данных в векторное хранилище

Основные компоненты:
- config.py: Конфигурация ETL процесса
- pipeline.py: Основной пайплайн обработки
- main.py: Точка входа для запуска ETL
- core/: Ядро ETL процесса (коннекторы, трансформеры, загрузчики)
- factory/: Фабрики для создания компонентов
- schema/: Управление схемой базы данных
- splitters/: Разделители текста на чанки
- metadata/: Генераторы метаданных

Поддерживаемые режимы работы:
- Синхронный режим (SyncETLRunner)
- Асинхронный режим (AsyncETLRunner)
- Различные провайдеры эмбеддингов (Sentence Transformers, BERT)
- Различные бэкенды хранения (PostgreSQL/pgvector, FAISS)
"""

from .config import settings
from .pipeline import ETLPipeline
from .main import main

__all__ = [
    "settings",
    "ETLPipeline",
    "main"
]