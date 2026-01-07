"""
Модуль для разбиения текста на чанки в VectorDB.

Предоставляет различные алгоритмы разделения текста на семантически осмысленные части:
- sentence_splitter.py: Разделитель по предложениям с поддержкой русского языка
- semantic_chunker.py: Семантический разделитель на основе эмбеддингов
- base.py: Абстрактный базовый класс для всех разделителей

Ключевые алгоритмы:
- SentenceSplitter: Использует правила языка и пунктуацию для разделения по предложениям
- SemanticChunker: Использует косинусное сходство между эмбеддингами для определения границ чанков

Параметры настройки:
- threshold: Порог сходства для SemanticChunker (по умолчанию 0.4)
- min_chunk_size: Минимальный размер чанка в предложениях
- abbreviations: Список аббревиатур для корректного разделения

Использование:

from etl.core.splitters import SentenceSplitter, SemanticChunker

# Простое разделение по предложениям
sentence_splitter = SentenceSplitter()
sentences = sentence_splitter.split(text)

# Семантическое разделение
semantic_chunker = SemanticChunker(embedder, sentence_splitter)
chunks = semantic_chunker.split(text)

Этот модуль критически важен для качества поиска, так как определяет,
как текст будет разбит на части для векторизации и поиска.
"""
__all__ = ["SentenceSplitter"]
