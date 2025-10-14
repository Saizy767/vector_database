import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncConnection
from shared.models import Base

logger = logging.getLogger(__name__)


class AsyncSchemaManager:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def ensure_vector_extension(self):
        async with self.engine.begin() as conn:
            conn: AsyncConnection
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("✅ Расширение 'vector' доступно.")
            except Exception as e:
                logger.error(f"❌ Не удалось создать расширение 'vector': {e}")
                raise

    async def create_tables(self):
        logger.info("Создание таблиц...")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Таблицы созданы.")

    async def initialize(self):
        await self.ensure_vector_extension()
        await self.create_tables()