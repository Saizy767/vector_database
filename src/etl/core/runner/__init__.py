"""
Модуль запуска ETL-процессов (Extract, Transform, Load) для VectorDB.

Этот пакет содержит реализации различных стратегий выполнения ETL-пайплайнов:
- Синхронный режим (SyncETLRunner) для простых сценариев и отладки
- Асинхронный режим (AsyncETLRunner) для высокопроизводительной обработки
- Абстрактный интерфейс (IETLRunner) для обеспечения единообразия

Структура пакета:
├── abc.py             # Абстрактный базовый класс IETLRunner
├── sync_runner.py     # Синхронная реализация ETL-процесса
└── async_runner.py    # Асинхронная реализация ETL-процесса с поддержкой asyncio

Ключевые особенности:
- Единый интерфейс для разных режимов выполнения
- Поддержка инициализации, выполнения и корректного завершения
- Встроенная обработка ошибок и управление состоянием
- Гибкость в настройке компонентов через фабрики

Принцип работы:
1.	initialize() - настройка соединений, создание таблиц, подготовка индексов
2.	run() - выполнение основного ETL-пайплайна
3.	shutdown() - освобождение ресурсов и сохранение состояния

Выбор режима выполнения определяется через конфигурацию (async_mode в .env файле).
Асинхронный режим рекомендуется для production-окружений с большими объемами данных.

Пример использования:

from etl.core.runner import SyncETLRunner, AsyncETLRunner
from etl.config import ETLSettings

settings = ETLSettings()
runner = AsyncETLRunner(settings) if settings.async_mode else SyncETLRunner(settings)

await runner.initialize()
await runner.run()
await runner.shutdown()

Этот модуль является ключевым компонентом ETL-архитектуры, обеспечивая гибкость
и масштабируемость процессов обработки данных в VectorDB.
"""

from .abc import IETLRunner
from .sync_runner import SyncETLRunner
from .async_runner import AsyncETLRunner

__all__ = [
    "IETLRunner",
    "SyncETLRunner",
    "AsyncETLRunner"
]