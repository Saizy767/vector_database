import logging

from typing import Optional, List
from etl.core.etl.extractors.sql_extractor import SQLExtractor
from etl.core.etl.transformers.transformer import Transformer
from etl.core.etl.loaders.sql_loader import SQLLoader
from etl.core.etl.base import BaseExtractor, BaseLoader, BaseTransformer


logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(
        self,
        extractor: BaseExtractor,
        transformer: BaseTransformer,
        loader: BaseLoader,
        batch_size: int = 100,
    ):
        """
        Универсальный VectorDB, поддерживающий любой loader (SQL, FAISS и т.д.).
        """
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.batch_size = batch_size
        logger.info("Intialized VectorDB with extractor, transformer, loader")

    def transform_table(
        self,
        source_table: str,
        text_column: str,
        source_id_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        if not text_column:
            raise ValueError("text_column is not defined")
        logger.info(f"Starting transform from '{source_table}' using {type(self.loader).__name__}")

        for batch in self.extractor.extract_batches(
            table_name=source_table,
            batch_size=self.batch_size,
            columns=columns,
        ):
            if not batch:
                continue
            transformed_batches = self.transformer.transform(
                batch_rows=batch,
                text_column=text_column,
                source_id_column=source_id_column,
            )
            for transformed_batch in transformed_batches:
                if transformed_batch:
                    self.loader.load(transformed_batch)

        logger.info("Transform and load completed")