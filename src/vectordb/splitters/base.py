from abc import ABC, abstractmethod
from typing import List

class BaseSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass