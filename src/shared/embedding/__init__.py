"""
Модуль для создания эмбеддингов текста в VectorDB.

Предоставляет унифицированный интерфейс для различных моделей эмбеддингов:
- sentence_transformer.py: Модели на основе Sentence Transformers
- bert.py: Модели на основе BERT архитектуры
- base.py: Абстрактный базовый класс для всех эмбеддеров

Ключевые возможности:
- Поддержка различных моделей (all-MiniLM-L6-v2, bert-base-uncased и др.)
- Автоматическое определение устройства (CPU/GPU)
- Нормализация векторов для поиска по косинусному сходству
- Кэширование для ускорения повторных вычислений
- Обработка больших текстов через пакетную обработку

Использование:

from shared.embedding import SentenceTransformerEmbedding

embedder = SentenceTransformerEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # или "cpu"
)
embedding = embedder.embed_text("Текст для векторизации")

Этот модуль используется как в ETL пайплайне для индексации данных,
так и в поисковом сервисе для векторизации запросов.
"""

from .sentence_transformer import SentenceTransformerEmbedding
from .bert import BERTEmbedder

__all__ = [
    "SentenceTransformerEmbedding",
    "BERTEmbedder"
    ]
