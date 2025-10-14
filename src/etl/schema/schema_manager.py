import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine
from shared import Base

logger = logging.getLogger(__name__)


class SchemaManager:
    def __init__(self, engine: Engine):
        self.engine = engine

    def ensure_vector_extension(self):
        with self.engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("✅ Расширение 'vector' доступно.")
            except Exception as e:
                logger.error(f"❌ Не удалось создать расширение 'vector': {e}")
                raise

    def create_tables(self):
        logger.info("Создание таблиц...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ Таблицы созданы.")

    def initialize(self):
        self.ensure_vector_extension()
        self.create_tables()