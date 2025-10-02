from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
        Базовый класс для всех Extractor
    """
    @abstractmethod
    def extract_all(self):
        pass

    @abstractmethod    
    def extract_batch(self):
        pass


class BaseTransformer(ABC):
    """
        Базовый класс для всех Transformer
    """
    @abstractmethod
    def transform(self):
        pass


class BaseLoader(ABC):
    """
        Базовый класс для всех Loader
    """
    @abstractmethod
    def load(self):
        pass