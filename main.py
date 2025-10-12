# main.py
import logging
from settings import settings
from vectordb import (
    VectorDB,
    SQLConnector,
    SentenceTransformerEmbedding,
    SentenceSpliter,
    MetadataBuilder,
    BERTEmbedder
)
from models import EmbeddingChapter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()  # опционально: оставить и консольный вывод
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Запуск VectorDB ETL-пайплайна")

    
    connector = SQLConnector(settings.db_url)


    if settings.embedding_provider == "sentence-transformers":
        embedding = SentenceTransformerEmbedding(
                model_name=settings.embedding_model,
                device=settings.device,
            )
    elif settings.embedding_provider == "bert":
        embedding = BERTEmbedder(
                model_name=settings.embedding_model,
                device=settings.device,
            )
    else:
        raise NotImplementedError(f"Provider '{settings.embedding_provider}' not supported yet")
    

    
    splitter = SentenceSpliter()

    
    field_mapping = {col: col for col in settings.metadata_columns} if settings.metadata_columns else {}
    metadata_builder = MetadataBuilder(field_mapping=field_mapping)

    
    vdb = VectorDB(
        connector=connector,
        embedding=embedding,
        splitter=splitter,
        metadata_builder=metadata_builder,
        target_table=settings.load_table_name,
        batch_size=100,
        orm_class=EmbeddingChapter
    )
    if not settings.embedding_columns:
        raise ValueError("(embedding_columns) must contain at least one column name in .env")
    text_column = settings.embedding_columns[0]

    # 6. Запуск трансформации
    vdb.transform_table(
        source_table=settings.extract_table_name,
        text_column= text_column,
        source_id_column=settings.source_id,
        columns=(settings.embedding_columns + (settings.metadata_columns or [])),
    )

    logger.info("✅ ETL-пайплайн завершён успешно.")


if __name__ == "__main__":
    main()