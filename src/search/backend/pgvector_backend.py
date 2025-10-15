import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from search.core.abc import BaseSearchBackend

logger = logging.getLogger(__name__)

class PGVectorBackend(BaseSearchBackend):
    def __init__(self, db_url: str):
        if not db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None

    async def connect(self):
        self.engine = create_async_engine(self.db_url, echo=False)
        self.SessionLocal = async_sessionmaker(self.engine, expire_on_commit=False)
        logger.info("✅ PGVectorBackend connected")

    async def disconnect(self):
        if self.engine:
            await self.engine.dispose()
            logger.info("🔌 PGVectorBackend disconnected")

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
        async with self.SessionLocal() as session:
            vector_str = "[" + ",".join(f"{x:.6f}" for x in query_vector) + "]"

            where_parts = []
            params = {
                "query_vector": vector_str,
                "top_k": top_k,
                "dist_thresh": 2.0 * (1.0 - min_similarity)
            }

            if metadata_filter:
                for i, (key, value) in enumerate(metadata_filter.items()):
                    param_name = f"meta_val_{i}"
                    if key.startswith("data."):
                        json_path = key[5:]
                        where_parts.append(f"(metadata->'data'->>'{json_path}') = :{param_name}")
                    else:
                        where_parts.append(f"(metadata->>'{key}') = :{param_name}")
                    params[param_name] = str(value)

            where_clause = " AND ".join(where_parts)
            if where_clause:
                where_clause = "WHERE " + where_clause + " AND"

            sql = f"""
                SELECT id, chunk_text, embedding, metadata,
                       embedding <=> :query_vector AS distance
                FROM embedding_chapter
                {where_clause} (embedding <=> :query_vector) <= :dist_thresh
                ORDER BY embedding <=> :query_vector
                LIMIT :top_k
            """

            result = await session.execute(text(sql), params)
            rows = result.fetchall()

            results = []
            for row in rows:
                similarity = max(0.0, 1.0 - (row.distance / 2.0))
                results.append({
                    "id": row.id,
                    "chunk_text": row.chunk_text,
                    "metadata": row.metadata,
                    "similarity": float(similarity)
                })

            logger.debug(f"PGVector search returned {len(results)} results")
            return results