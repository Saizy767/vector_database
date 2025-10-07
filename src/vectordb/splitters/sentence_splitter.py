import re
from itertools import zip_longest
from typing import List
from .base import BaseSplitter

class SentenceSpliter(BaseSplitter):
    """
    Универсальный и масштабируемый сплиттер предложений.
    Без использования lookbehind (совместим с Python 3.13+).
    """

    def __init__(self, abbreviations: List[str] | None = None):
        # Дефолтные сокращения
        self.abbreviations = abbreviations or [
            "г", "ул", "рис", "стр", "т", "т.д", "т.п", "см", "им",
            "и.т.д", "и.п", "с", "д", "инж", "акад", "ред", "чл-кор"
        ]
        # Предкомпилированный паттерн сокращений
        self.abbr_regex = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, self.abbreviations)) + r')\.$',
            re.IGNORECASE
        )

    def normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("…", ".")
        return text.strip()

    def split(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        parts = re.split(r'([.!?]["»”\']?\s+)', text)
        if not parts:
            return []

        sentences = []
        buffer: list[str] = []

        for text_chunk, separator in zip_longest(parts[::2], parts[1::2], fillvalue=''):
            buffer.append(text_chunk)
            buffer.append(separator)

            current_sentence = ''.join(buffer).strip()

            if not self.abbr_regex.search(current_sentence):
                sentences.append(current_sentence)
                buffer.clear()
        if buffer:
            sentences.append(''.join(buffer).strip())

        return sentences