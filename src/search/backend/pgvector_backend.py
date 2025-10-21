import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from search.core.abc import BaseSearchBackend
from shared.models import create_embedding_model, Base
from shared.db_utils import get_embedding_dim_from_db  # ← новая утилита

logger = logging.getLogger(__name__)

class PGVectorBackend(BaseSearchBackend):
    def __init__(self, db_url: str, table_name: str):
        if not db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.db_url = db_url
        self.table_name = table_name

        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

        # Пока модель не создана — EmbeddingChapter = None
        self.EmbeddingChapter = None

    async def initialize(self, expected_dim: int):
        """
        Вызывается один раз при старте.
        1. Определяет фактическую размерность из БД.
        2. Сравнивает с expected_dim (от эмбеддера).
        3. Создаёт ORM-модель.
        """
        actual_dim = await get_embedding_dim_from_db(self.engine, self.table_name)
        if actual_dim != expected_dim:
            raise ValueError(
                f"Несовпадение размерности эмбеддингов:\n"
                f"  - В БД (таблица {self.table_name}): {actual_dim}\n"
                f"  - У поискового эмбеддера: {expected_dim}\n"
                f"Проверьте, что ETL и Search используют одну и ту же модель."
            )

        self.EmbeddingChapter = create_embedding_model(
            dim=actual_dim,
            table_name=self.table_name
        )
        logger.info(f"✅ EmbeddingChapter создан с dim={actual_dim} для таблицы {self.table_name}")

    async def connect(self):
        pass

    async def disconnect(self):
        await self.engine.dispose()
        logger.info("🔌 PGVectorBackend engine disposed")

    async def health_check(self) -> bool:
        try:
            async with self.SessionLocal() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.EmbeddingChapter is None:
            raise RuntimeError("PGVectorBackend не инициализирован. Вызовите .initialize()")

        from sqlalchemy import and_
        async with self.SessionLocal() as session:
            dist_thresh = 2.0 * (1.0 - min_similarity)
            stmt = (
                select(
                    self.EmbeddingChapter.id,
                    self.EmbeddingChapter.chunk_text,
                    self.EmbeddingChapter.embedding,
                    self.EmbeddingChapter.metadata_,
                    self.EmbeddingChapter.embedding.cosine_distance(query_vector).label("distance")
                )
                .where(self.EmbeddingChapter.embedding.cosine_distance(query_vector) <= dist_thresh)
                .order_by(self.EmbeddingChapter.embedding.cosine_distance(query_vector))
                .limit(top_k)
            )

            if metadata_filter:
                from sqlalchemy import text as sql_text
                where_clauses = []
                params = {}
                for key, value in metadata_filter.items():
                    param_name = f"val_{key.replace('.', '_')}"
                    if key.startswith("data."):
                        json_path = key[5:]
                        where_clauses.append(
                            sql_text(f"(metadata_->'data'->>'{json_path}') = :{param_name}")
                        )
                    else:
                        where_clauses.append(
                            sql_text(f"(metadata_->>'{key}') = :{param_name}")
                        )
                    params[param_name] = str(value)
                stmt = stmt.where(and_(*where_clauses))
            else:
                params = {}

            result = await session.execute(stmt, params)
            rows = result.fetchall()
            results = []
            for row in rows:
                similarity = max(0.0, 1.0 - (row.distance / 2.0))
                results.append({
                    "id": row.id,
                    "chunk_text": row.chunk_text,
                    "metadata": row.metadata_,
                    "similarity": float(similarity)
                })
            logger.debug(f"PGVector search returned {len(results)} results")
            return results