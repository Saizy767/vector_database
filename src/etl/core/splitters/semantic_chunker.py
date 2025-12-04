import logging
from typing import List, Optional
from .base import BaseSplitter
from shared.embedding.base import BaseEmbedding
from shared.utils import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker(BaseSplitter):
    def __init__(
        self,
        embedder: BaseEmbedding,
        sentence_splitter: Optional[BaseSplitter] = None,
        threshold: float = 0.4,
        min_chunk_size: int = 1,
    ):
        self.embedder = embedder
        self.sentence_splitter = sentence_splitter
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        logger.info(f"SemanticChunker initialized with threshold={threshold}")

    def normalize_text(self, text: str) -> str:
        return self.sentence_splitter.normalize_text(text)

    def split(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        if not text.strip():
            return []

        sentences = self.sentence_splitter.split(text)
        if len(sentences) == 0:
            return [text] if text else []
        if len(sentences) == 1:
            return sentences

        try:
            embeddings = [self.embedder.embed_text(sent) for sent in sentences]
        except Exception as e:
            logger.warning(f"Failed to embed sentences: {e}. Falling back to sentence splitting.")
            return sentences

        distances = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            sim = max(0.0, min(1.0, sim))
            dist = 1.0 - sim
            distances.append(dist)

        chunk_boundaries = [0]
        for i, dist in enumerate(distances):
            if dist > self.threshold:
                chunk_boundaries.append(i + 1)
        chunk_boundaries.append(len(sentences))

        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            if len(chunk_sentences) >= self.min_chunk_size:
                chunk = " ".join(chunk_sentences)
                chunks.append(chunk.strip())
            else:
                if chunks:
                    chunks[-1] += " " + " ".join(chunk_sentences)
                else:
                    chunks.append(" ".join(chunk_sentences))

        logger.debug(
            f"SemanticChunker: {len(sentences)} sentences → {len(chunks)} chunks (threshold={self.threshold})"
        )
        return chunks