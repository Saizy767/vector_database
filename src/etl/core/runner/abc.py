from abc import ABC, abstractmethod
from typing import Optional
from etl.config import ETLSettings

class IETLRunner(ABC):
    def __init__(self, settings: ETLSettings):
        self.settings = settings
        self._status: str = "initialized"  # Например: "initialized", "running", "completed", "failed", "shutdown"

    @property
    def status(self) -> str:
        return self._status

    async def set_status(self, status: str) -> None:
        self._status = status

    @abstractmethod
    async def initialize(self) -> None:
        """
        Инициализация ресурсов: подключение к БД, проверка схемы, создание индексов и т.д.
        Выполняется **до** основного ETL.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """
        Основной ETL-процесс: extract → transform → load.
        Должен вызывать `initialize()` при необходимости или ожидать, что он вызван заранее.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Корректное завершение: закрытие соединений, сохранение индексов (например, FAISS),
        освобождение ресурсов.
        """
        pass