import logging
import asyncio
from typing import Optional, List, Dict, Any, Generator
from etl.core.etl.base import BaseTransformer
from shared.embedding.base import BaseEmbedding
from etl.core.splitters.base import BaseSplitter
from etl.core.metadata.metadata_builder import MetadataBuilder

logger = logging.getLogger(__name__)

class Transformer(BaseTransformer):
    def __init__(
        self,
        embedding: BaseEmbedding,
        splitter: BaseSplitter,
        metadata_builder: Optional[MetadataBuilder] = None,
        metadata_columns: Optional[List[str]] = None
    ):
        self.embedding = embedding
        self.splitter = splitter
        self.metadata_builder = metadata_builder or MetadataBuilder()
        self.metadata_columns = metadata_columns or []
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

            metadata_row = {}
            if self.metadata_columns:
                for col in self.metadata_columns:
                    if col in row:
                        metadata_row[col] = row[col]
            else:
                metadata_row = {}

            chunks = self.splitter.split(raw_text)
            total_chunks = len(chunks)
            source_id = str(row[source_id_column]) if source_id_column and source_id_column in row else None

            logger.debug(f"Row {row_idx}: split into {total_chunks} chunks")
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                embedding = self.embedding.embed_text(chunk)
                
                metadata = self.metadata_builder.build(
                    row_data=metadata_row,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    source_id=source_id,
                )
                results.append({
                    "chunk_text": chunk,
                    "embedding": embedding,
                    "metadata_": metadata,
                })
        logger.info(f"Produced {len(results)} transformed chunks")
        yield results

    async def atransform(
        self,
        batch_rows: List[Dict[str, Any]],
        text_column: str = "",
        source_id_column: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        logger.info(f"Async-transforming batch of {len(batch_rows)} rows")

        loop = asyncio.get_running_loop()
        tasks = []

        for row_idx, row in enumerate(batch_rows):
            raw_text = row.get(text_column)
            if not raw_text or not isinstance(raw_text, str):
                logger.warning(
                    f"Row {row_idx}: skipped. Value in '{text_column}' = {repr(raw_text)} (type: {type(raw_text).__name__})"
                )
                continue

            metadata_row = {
                col: row[col] for col in self.metadata_columns if col in row
            } if self.metadata_columns else {}

            chunks = self.splitter.split(raw_text)
            total_chunks = len(chunks)
            source_id = str(row[source_id_column]) if source_id_column and source_id_column in row else None
            logger.debug(f"Row {row_idx}: split into {total_chunks} chunks")

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                task = loop.run_in_executor(
                    None,
                    self._safe_embed_and_build,
                    chunk,
                    metadata_row,
                    idx,
                    total_chunks,
                    source_id,
                    row_idx
                )
                tasks.append(task)

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for res in completed:
            if isinstance(res, Exception):
                logger.error(f"Async transform task failed: {res}")
                continue
            if res is not None:
                results.append(res)

        logger.info(f"Async transformation produced {len(results)} chunks")
        return results
    

    def _safe_embed_and_build(
        self,
        chunk: str,
        metadata_row: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        source_id: Optional[str],
        row_idx: int
    ) -> Optional[Dict[str, Any]]:
        try:
            embedding = self.embedding.embed_text(chunk)
            metadata = self.metadata_builder.build(
                row_data=metadata_row,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                source_id=source_id,
            )
            return {
                "chunk_text": chunk,
                "embedding": embedding,
                "metadata_": metadata,
            }
        except Exception as e:
            logger.error(f"Failed to embed chunk in row {row_idx}, index {chunk_index}: {e}")
            return None