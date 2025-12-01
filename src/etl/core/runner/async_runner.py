import logging
from etl.core.runner.abc import IETLRunner
from etl.factory.async_factory import AsyncComponentFactory
from etl.schema.async_schema_manager import AsyncSchemaManager
from etl.core.async_vector_db import AsyncVectorDB
logger = logging.getLogger(__name__)

class AsyncETLRunner(IETLRunner):
    def __init__(self, settings):
        super().__init__(settings)
        self.factory = None
        self.connector = None
        self.embedder = None
        self.orm_model = None
        self.transformer = None
        self.extractor = None
        self.loader = None
        self.vdb = None

    async def initialize(self) -> None:
        """Инициализация компонентов и схемы базы данных."""
        logger.info("Initializing AsyncETLRunner components...")
        self.factory = AsyncComponentFactory(self.settings)

        self.connector = self.factory.create_connector()
        self.embedder = self.factory.create_embedder()
        embedding_dim = self.factory.get_embedding_dim(self.embedder)
        self.orm_model = self.factory.create_orm_model(embedding_dim)

        # Инициализация схемы БД
        schema_manager = AsyncSchemaManager(self.connector.engine)
        await schema_manager.initialize()

        # Создание остальных компонентов
        splitter = self.factory.create_splitter()
        metadata_builder = self.factory.create_metadata_builder()
        self.extractor = self.factory.create_extractor(self.connector)
        self.loader = self.factory.create_loader(orm_class=self.orm_model)

        self.transformer = self.factory.create_transformer(
            embedder=self.embedder,
            splitter=splitter,
            metadata_builder=metadata_builder,
        )

        self.vdb = AsyncVectorDB(
            extractor=self.extractor,
            transformer=self.transformer,
            loader=self.loader,
            batch_size=self.settings.batch_size,
        )
        logger.info("AsyncETLRunner initialized successfully.")

    async def run(self) -> None:
        """Запуск ETL-процесса после инициализации."""
        if not hasattr(self, 'vdb') or self.vdb is None:
            await self.initialize()  # fallback, но лучше вызывать явно

        text_column = self.factory.get_text_column()
        columns = self.factory.get_columns()
        logger.info(
            f"Begin async ETL: '{self.settings.extract_table_name}' → '{self.vdb.loader.orm_class.__tablename__}'"
        )

        await self.vdb.async_transform_table(
            source_table=self.settings.extract_table_name,
            text_column=text_column,
            source_id_column=self.settings.source_id,
            columns=columns,
        )
        logger.info("Async ETL-pipeline complete.")

    async def shutdown(self) -> None:
        """Завершение работы (опционально закрытие соединений)."""
        if self.connector:
            await self.connector.close()
        logger.info("AsyncETLRunner shut down.")