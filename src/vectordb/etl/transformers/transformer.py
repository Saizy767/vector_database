import logging
from typing import Optional, List, Dict, Any, Generator
from vectordb.etl.base import BaseTransformer
from vectordb.embedding.base import BaseEmbedding
from vectordb.splitters.base import BaseSplitter
from vectordb.metadata.metadata_builder import MetadataBuilder

logger = logging.getLogger(__name__)

class Transformer(BaseTransformer):
    def __init__(
        self,
        embedding: BaseEmbedding,
        splitter: BaseSplitter,
        metadata_builder: Optional[MetadataBuilder] = None,
    ):
        self.embedding = embedding
        self.splitter = splitter
        self.metadata_builder = metadata_builder or MetadataBuilder()
        logger.debug("Transformer initialized")

    def transform(
        self,
        batch_rows: List[Dict[str, Any]],
        text_column: str = "",
        source_id_column: Optional[str] = None,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        results = []
        logger.info(f"Transforming batch of {len(batch_rows)} rows")
        for row_idx, row in enumerate(batch_rows):
            raw_text = row.get(text_column)
            if not raw_text or not isinstance(raw_text, str):
                logger.warning(
                    f"Row {row_idx}: skipped. Value in '{text_column}' = {repr(raw_text)} (type: {type(raw_text).__name__})"
                )
                continue

            chunks = self.splitter.split(raw_text)
            total_chunks = len(chunks)
            source_id = str(row[source_id_column]) if source_id_column and source_id_column in row else None

            logger.debug(f"Row {row_idx}: split into {total_chunks} chunks")
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                embedding = self.embedding.embed_text(chunk)
                metadata = self.metadata_builder.build(
                    row_data=row,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    source_id=source_id,
                )
                results.append({
                    "chunk_text": chunk,
                    "embedding": embedding.tolist(),  # для сериализации
                    "metadata_": metadata,
                })
        logger.info(f"Produced {len(results)} transformed chunks")
        yield results