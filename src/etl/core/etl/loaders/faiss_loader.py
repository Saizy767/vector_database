import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from etl.core.etl.base import BaseLoader

logger = logging.getLogger(__name__)

class FAISSLoader(BaseLoader):
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_dim: int,
        faiss_index_type: str = "FlatIP",
        metadata_format: str = "jsonl"
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_dim = embedding_dim
        self.faiss_index_type = faiss_index_type
        self.metadata_format = metadata_format

        if self.metadata_format != "jsonl":
            raise NotImplementedError(f"Метаданные формата '{self.metadata_format}' пока не поддерживаются. Используйте 'jsonl'.")

        self._index: faiss.Index = None
        self._metadata_list: List[Dict[str, Any]] = []
        self._current_id = 0

        self._load_existing_data()

    def _create_index(self) -> faiss.Index:
        if self.faiss_index_type == "FlatIP":
            return faiss.IndexFlatIP(self.embedding_dim)
        elif self.faiss_index_type == "FlatL2":
            return faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise NotImplementedError(f"Тип индекса FAISS '{self.faiss_index_type}' пока не поддерживается.")

    def _load_existing_data(self):
        if self.index_path.exists():
            logger.info(f"Загрузка существующего FAISS индекса из {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            actual_dim = self._index.d
            if actual_dim != self.embedding_dim:
                raise ValueError(
                    f"Размерность существующего индекса ({actual_dim}) "
                    f"не совпадает с ожидаемой ({self.embedding_dim})."
                )
            logger.info(f"Индекс загружен, размерность: {actual_dim}, количество векторов: {self._index.ntotal}")
        else:
            logger.info(f"FAISS индекс {self.index_path} не найден, создается новый.")
            self._index = self._create_index()

        if self.metadata_path.exists():
            logger.info(f"Загрузка существующих метаданных из {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            self._metadata_list.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка при чтении JSON строки {line_num} в {self.metadata_path}: {e}")
                            raise
            logger.info(f"Загружено {len(self._metadata_list)} записей метаданных.")
            self._current_id = len(self._metadata_list)
        else:
            logger.info(f"Файл метаданных {self.metadata_path} не найден, будет создан.")
            self._current_id = 0

    def _ensure_normalized_for_ip(self, vectors: np.ndarray) -> np.ndarray:
        if self.faiss_index_type == "FlatIP":
            faiss.normalize_L2(vectors)
        return vectors

    def load(self, data: List[Dict[str, Any]]) -> None:
        if not data:
            logger.debug("Нет данных для загрузки в FAISS.")
            return

        logger.info(f"Загрузка {len(data)} записей в FAISS индекс и {self.metadata_path}.")

        first_embedding = data[0]['embedding']
        if isinstance(first_embedding, list):
             first_embedding = np.array(first_embedding)
        if first_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Размерность эмбеддинга в данных ({first_embedding.shape[0]}) "
                f"не совпадает с ожидаемой ({self.embedding_dim})."
            )

        embeddings_list = [item['embedding'] for item in data]
        embeddings_array = np.array(embeddings_list).astype('float32')
        embeddings_array = self._ensure_normalized_for_ip(embeddings_array)

        self._index.add(embeddings_array)
        logger.debug(f"Добавлено {len(embeddings_list)} векторов в индекс.")

        with open(self.metadata_path, 'a', encoding='utf-8') as f_meta:
            for item in data:
                meta_to_save = {
                    "chunk_text": item.get('chunk_text', ''),
                    "metadata": item.get('metadata_', {})
                }
                f_meta.write(json.dumps(meta_to_save, ensure_ascii=False) + '\n')
                self._metadata_list.append(meta_to_save)
                self._current_id += 1

        faiss.write_index(self._index, str(self.index_path))
        logger.info(f"FAISS индекс сохранен в {self.index_path}. Всего векторов: {self._index.ntotal}")
