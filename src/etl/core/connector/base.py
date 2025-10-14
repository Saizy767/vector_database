from abc import ABC, abstractmethod

class BaseConnector(ABC):
    """Абстрактный коннектор для любых источников данных."""
    
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass