# main.py
import asyncio
import logging
from settings import settings
from vectordb import (
    VectorDB,
    SQLConnector,
    SentenceTransformerEmbedding,
    SentenceSplitter,
    MetadataBuilder,
    BERTEmbedder,
    SchemaManager
)
from vectordb.async_vector_db import AsyncVectorDB
from vectordb.connector.async_sql_connector import AsyncSQLConnector
from vectordb.schema_manager.async_schema_manager import AsyncSchemaManager
from models import EmbeddingChapter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_embedding():
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            device=settings.device,
        )
    elif settings.embedding_provider == "bert":
        return BERTEmbedder(
            model_name=settings.embedding_model,
            device=settings.device,
        )
    else:
        raise NotImplementedError(f"Provider '{settings.embedding_provider}' not supported yet")


async def async_main():
    logger.info("🔄 Режим: ASYNC")
    connector = AsyncSQLConnector(settings.db_url)


    async_schema_manager = AsyncSchemaManager(connector.engine)
    await async_schema_manager.initialize()

    embedding = create_embedding()
    splitter = SentenceSplitter()
    field_mapping = {col: col for col in settings.metadata_columns} if settings.metadata_columns else {}
    metadata_builder = MetadataBuilder(field_mapping=field_mapping)

    if not settings.embedding_columns:
        raise ValueError("(embedding_columns) must contain at least one column name in .env")
    text_column = settings.embedding_columns[0]
    columns = (settings.embedding_columns + (settings.metadata_columns or []))

    vdb = AsyncVectorDB(
        connector=connector,
        embedding=embedding,
        splitter=splitter,
        metadata_builder=metadata_builder,
        target_table=settings.load_table_name,
        batch_size=5,
        orm_class=EmbeddingChapter,
        metadata_columns=settings.metadata_columns
    )

    logger.info(f"Starting async transform from '{settings.extract_table_name}' to '{vdb.target_table}'")
    async for batch in vdb.extractor.extract_batches(
        table_name=settings.extract_table_name,
        batch_size=vdb.batch_size,
        columns=columns,
    ):
        if not batch:
            continue
        transformed_chunks = await vdb.transformer.atransform(
            batch_rows=batch,
            text_column=text_column,
            source_id_column=settings.source_id,
        )
        if transformed_chunks:
            session = await connector.connect()
            async with session.begin():
                await vdb.loader.load(session, transformed_chunks)

    logger.info("✅ Async ETL-пайплайн завершён успешно.")


def sync_main():
    logger.info("🔄 Режим: SYNC")
    connector = SQLConnector(settings.db_url)
    schema_manager = SchemaManager(connector.engine)
    schema_manager.initialize()

    embedding = create_embedding()
    splitter = SentenceSplitter()
    field_mapping = {col: col for col in settings.metadata_columns} if settings.metadata_columns else {}
    metadata_builder = MetadataBuilder(field_mapping=field_mapping)

    if not settings.embedding_columns:
        raise ValueError("(embedding_columns) must contain at least one column name in .env")
    text_column = settings.embedding_columns[0]
    columns = (settings.embedding_columns + (settings.metadata_columns or []))

    vdb = VectorDB(
        connector=connector,
        embedding=embedding,
        splitter=splitter,
        metadata_builder=metadata_builder,
        target_table=settings.load_table_name,
        batch_size=20,
        orm_class=EmbeddingChapter,
        metadata_columns=settings.metadata_columns
    )

    logger.info(f"Starting sync transform from '{settings.extract_table_name}' to '{vdb.target_table}'")
    for batch in vdb.extractor.extract_batches(
        table_name=settings.extract_table_name,
        batch_size=vdb.batch_size,
        columns=columns,
    ):
        transformed_batches = vdb.transformer.transform(
            batch_rows=batch,
            text_column=text_column,
            source_id_column=settings.source_id,
        )
        for transformed_batch in transformed_batches:
            if transformed_batch:
                vdb.loader.load(transformed_batch)

    logger.info("✅ Sync ETL-пайплайн завершён успешно.")


def main():
    logger.info("🚀 Запуск VectorDB ETL-пайплайна")
    if settings.async_mode:
        asyncio.run(async_main())
    else:
        sync_main()


if __name__ == "__main__":
    main()