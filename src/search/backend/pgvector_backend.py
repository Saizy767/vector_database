import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from search.core.abc import BaseSearchBackend
from shared.models import EmbeddingChapter
from shared.utils import deserialize_json_field

logger = logging.getLogger(__name__)

class PGVectorBackend(BaseSearchBackend):
    def __init__(self, db_url: str):
        if not db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.db_url = db_url
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
        logger.info("✅ PGVectorBackend initialized with connection pool")

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
        from sqlalchemy import and_

        async with self.SessionLocal() as session:
            # Вычисляем порог расстояния (cosine distance)
            dist_thresh = 2.0 * (1.0 - min_similarity)

            # Базовый запрос: ближайшие векторы
            stmt = (
                select(
                    EmbeddingChapter.id,
                    EmbeddingChapter.chunk_text,
                    EmbeddingChapter.embedding,
                    EmbeddingChapter.metadata_,  # ← это поле автоматически десериализуется в dict
                    EmbeddingChapter.embedding.cosine_distance(query_vector).label("distance")
                )
                .where(EmbeddingChapter.embedding.cosine_distance(query_vector) <= dist_thresh)
                .order_by(EmbeddingChapter.embedding.cosine_distance(query_vector))
                .limit(top_k)
            )

            # Применяем фильтрацию по метаданным (если есть)
            if metadata_filter:
                from sqlalchemy import text
                where_clauses = []
                for key, value in metadata_filter.items():
                    if key.startswith("data."):
                        json_path = key[5:]
                        where_clauses.append(
                            text(f"(metadata_->'data'->>'{json_path}') = :val_{key.replace('.', '_')}")
                        )
                    else:
                        where_clauses.append(
                            text(f"(metadata_->>'{key}') = :val_{key.replace('.', '_')}")
                        )
                stmt = stmt.where(and_(*where_clauses))

            # Выполняем запрос
            result = await session.execute(stmt, {
                f"val_{k.replace('.', '_')}": str(v) for k, v in metadata_filter.items()
            } if metadata_filter else {})

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