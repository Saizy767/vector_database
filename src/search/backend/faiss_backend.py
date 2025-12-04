import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from search.core.abc import BaseSearchBackend

logger = logging.getLogger(__name__)


class FAISSBackend(BaseSearchBackend):
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS индекс не найден: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {self.metadata_path}")

        self.index: faiss.Index = faiss.read_index(str(self.index_path))
        self.dim = self.index.d
        self.ntotal = self.index.ntotal

        self.metadata_list: List[Dict[str, Any]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata_list.append(json.loads(line))

        if len(self.metadata_list) != self.ntotal:
            raise ValueError(
                f"Несовпадение: FAISS содержит {self.ntotal} векторов, "
                f"но метаданных — {len(self.metadata_list)}."
            )

        logger.info(f"✅ FAISSBackend загружен: {self.ntotal} записей, dim={self.dim}")

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self.index = None
        self.metadata_list.clear()
        logger.info("🔌 FAISSBackend выгружен из памяти")

    async def health_check(self) -> bool:
        return self.index is not None and len(self.metadata_list) == self.ntotal

    async def initialize(self, expected_dim: int) -> None:
        """
        Проверяет соответствие размерности FAISS-индекса эмбеддеру.
        Вызывается из lifespan в app.py.
        """
        if self.dim != expected_dim:
            raise ValueError(
                f"Размерность FAISS-индекса ({self.dim}) не совпадает с эмбеддером ({expected_dim})."
            )
        logger.info(f"✅ FAISSBackend прошёл проверку размерности: {expected_dim}")

    def _apply_metadata_filter(self, candidates: List[Dict], metadata_filter: Dict[str, Any]) -> List[Dict]:
        """
        Применяет in-memory фильтрацию по метаданным.
        Поддерживает вложенные ключи через точку, например: "data.author".
        """
        def get_nested_value(obj: Dict, key_path: str):
            keys = key_path.split(".")
            for k in keys:
                if isinstance(obj, dict) and k in obj:
                    obj = obj[k]
                else:
                    return None
            return obj

        filtered = []
        for item in candidates:
            match = True
            for key, expected_val in metadata_filter.items():
                actual_val = get_nested_value(item["metadata"], key)
                if actual_val != expected_val:
                    match = False
                    break
            if match:
                filtered.append(item)
        return filtered

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_similarity: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISSBackend не загружен")

        q_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(q_vec)

        distances, indices = self.index.search(q_vec, top_k * 5 if metadata_filter else top_k)
        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue

            meta = self.metadata_list[idx]
            chunk_text = meta.get("chunk_text", "")
            metadata = meta.get("metadata", {})

            if isinstance(self.index, faiss.IndexFlatIP):
                similarity = float(np.clip(dist, 0.0, 1.0))
            elif isinstance(self.index, faiss.IndexFlatL2):
                similarity = 1.0 / (1.0 + float(dist))
            else:
                similarity = float(np.clip(1.0 - dist / 2.0, 0.0, 1.0))

            if similarity < min_similarity:
                continue

            results.append({
                "id": int(idx),
                "chunk_text": chunk_text,
                "metadata": metadata,
                "similarity": similarity
            })

        if metadata_filter:
            results = self._apply_metadata_filter(results, metadata_filter)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]