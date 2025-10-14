import logging

from typing import Optional, List
from .connector.sql_connector import SQLConnector
from etl.core.etl.extractors.sql_extractor import SQLExtractor
from etl.core.etl.transformers.transformer import Transformer
from etl.core.etl.loaders.sql_loader import SQLLoader
from etl.core.embedding.base import BaseEmbedding
from etl.core.splitters.base import BaseSplitter
from etl.core.metadata.metadata_builder import MetadataBuilder


logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(
        self,
        connector: SQLConnector,
        embedding: BaseEmbedding,
        splitter: BaseSplitter,
        metadata_builder: Optional[MetadataBuilder] = None,
        target_table: Optional[str] = None,
        batch_size: int = 100,
        orm_class = None,
        metadata_columns: Optional[List[str]] = None
    ):
        self.connector = connector
        self.batch_size = batch_size
        self.target_table = target_table

        self.extractor = SQLExtractor(connector)
        self.transformer = Transformer(
            embedding=embedding,
            splitter=splitter,
            metadata_builder=metadata_builder,
            metadata_columns=metadata_columns
        )
        self.loader = SQLLoader(
            connector=connector,
            table_name=target_table,
            batch_size=batch_size,
            orm_class=orm_class
        )
        logger.info("Intialized VectorDB with extractor, transformer, loader")

    def transform_table(
        self,
        source_table: str,
        text_column: Optional[str] = None,
        source_id_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        if not text_column:
            raise ValueError("text_column is not defined")
        if not self.target_table:
            raise ValueError("target_table must be specified for loading results.")
        
        logger.info(f"Starting transform from '{source_table}' to '{self.target_table}'")
        for batch in self.extractor.extract_batches(
            table_name=source_table,
            batch_size=self.batch_size,
            columns=columns,
        ):
            transformed_batches = self.transformer.transform(
                batch_rows=batch,
                text_column=text_column,
                source_id_column=source_id_column,
            )

            for transformed_batch in transformed_batches:
                if transformed_batch:
                    self.loader.load(transformed_batch)
        logger.info("Transform and load completed")