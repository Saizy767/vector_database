import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
import numpy as np
import faiss

from etl.core.etl.base import BaseExtractor

logger = logging.getLogger(__name__)

class FAISSFileExtractor(BaseExtractor):
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_as_list: bool = True
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_as_list = embedding_as_list

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.dim = self.index.d
        self.ntotal = self.index.ntotal

        self.metadata_list: List[Dict[str, Any]] = []
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata_list.append(json.loads(line))

        if len(self.metadata_list) != self.ntotal:
            raise ValueError(
                f"Несовпадение: FAISS содержит {self.ntotal} векторов, "
                f"но метаданных — {len(self.metadata_list)}."
            )

        logger.info(f"FAISSFileExtractor готов: {self.ntotal} записей загружено.")

    def extract_all(self) -> List[Dict[str, Any]]:
        """Возвращает все записи с векторами и метаданными."""
        vectors = self.index.reconstruct_n(0, self.ntotal)
        results = []
        for i in range(self.ntotal):
            meta = self.metadata_list[i]
            emb = vectors[i]
            if self.embedding_as_list:
                emb = emb.tolist()
            results.append({
                "id": i,
                "chunk_text": meta.get("chunk_text", ""),
                "embedding": emb,
                "metadata_": meta.get("metadata", {})
            })
        return results

    def extract_batches(self, batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """Генератор по батчам — экономия памяти при большом индексе."""
        vectors = self.index.reconstruct_n(0, self.ntotal)
        for i in range(0, self.ntotal, batch_size):
            batch = []
            for j in range(i, min(i + batch_size, self.ntotal)):
                meta = self.metadata_list[j]
                emb = vectors[j]
                if self.embedding_as_list:
                    emb = emb.tolist()
                batch.append({
                    "id": j,
                    "chunk_text": meta.get("chunk_text", ""),
                    "embedding": emb,
                    "metadata_": meta.get("metadata", {})
                })
            yield batch