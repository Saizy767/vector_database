import logging
from etl.core.runner.abc import IETLRunner
from etl.factory.sync_factory import SyncComponentFactory
from etl.schema.schema_manager import SchemaManager
from etl.core.vector_db import VectorDB
logger = logging.getLogger(__name__)

class SyncETLRunner(IETLRunner):
    def __init__(self, settings):
        super().__init__(settings)
        self.factory = None
        self.connector = None
        self.vdb = None

    async def initialize(self) -> None:
        """Инициализация компонентов и схемы базы данных (синхронная)."""
        logger.info("Initializing SyncETLRunner components...")
        self.factory = SyncComponentFactory(self.settings)

        self.connector = self.factory.create_connector()
        embedder = self.factory.create_embedder()
        embedding_dim = self.factory.get_embedding_dim(embedder)
        orm_model = self.factory.create_orm_model(embedding_dim)

        # Инициализация схемы
        schema_manager = SchemaManager(self.connector.engine)
        schema_manager.initialize()

        # Создание компонентов
        splitter = self.factory.create_splitter()
        metadata_builder = self.factory.create_metadata_builder()
        extractor = self.factory.create_extractor(self.connector)
        loader = self.factory.create_loader(connector=self.connector, orm_class=orm_model)

        transformer = self.factory.create_transformer(
            embedder=embedder,
            splitter=splitter,
            metadata_builder=metadata_builder,
        )

        self.vdb = VectorDB(
            extractor=extractor,
            transformer=transformer,
            loader=loader,
            batch_size=self.settings.batch_size,
        )
        logger.info("SyncETLRunner initialized successfully.")

    async def run(self) -> None:
        """Запуск синхронного ETL-процесса."""
        if self.vdb is None:
            await self.initialize()

        text_column = self.factory.get_text_column()
        columns = self.factory.get_columns()
        logger.info(
            f"Begin sync ETL: '{self.settings.extract_table_name}' → '{self.vdb.loader.table_name}'"
        )

        self.vdb.transform_table(
            source_table=self.settings.extract_table_name,
            text_column=text_column,
            source_id_column=self.settings.source_id,
            columns=columns,
        )
        logger.info("Sync ETL-pipeline complete.")

    async def shutdown(self) -> None:
        """Завершение работы."""
        if self.connector:
            self.connector.close()
        logger.info("SyncETLRunner shut down.")