import re
import logging
from itertools import zip_longest
from typing import List
from .base import BaseSplitter

logger = logging.getLogger(__name__)

class SentenceSpliter(BaseSplitter):
    def __init__(self, abbreviations: List[str] | None = None):
        self.abbreviations = abbreviations or [
            "г", "ул", "рис", "стр", "т", "т.д", "т.п", "см", "им",
            "и.т.д", "и.п", "с", "д", "инж", "акад", "ред", "чл-кор"
        ]
        self.abbr_regex = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, self.abbreviations)) + r')\.$',
            re.IGNORECASE
        )
        logger.debug("SentenceSpliter initialized")

    def normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("…", ".")
        return text.strip()

    def split(self, text: str) -> List[str]:
        original_len = len(text)
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
        logger.debug(f"Split text (len={original_len}) into {len(sentences)} sentences")
        return sentences