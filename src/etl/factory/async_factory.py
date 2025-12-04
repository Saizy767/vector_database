from etl.factory.base_factory import BaseComponentFactory
from etl.core.connector.async_sql_connector import AsyncSQLConnector
from etl.core.splitters.sentence_splitter import SentenceSplitter
from etl.core.metadata.metadata_builder import MetadataBuilder
from etl.core.etl.extractors.async_sql_extractor import AsyncSQLExtractor
from etl.core.etl.loaders.async_sql_loader import AsyncSQLLoader
from etl.core.etl.transformers.transformer import Transformer
from shared.embedding.sentence_transformer import SentenceTransformerEmbedding
from shared.embedding.bert import BERTEmbedder
from shared.models import create_embedding_model
from etl.core.splitters.semantic_chunker import SemanticChunker


class AsyncComponentFactory(BaseComponentFactory):
    def create_connector(self):
        return AsyncSQLConnector(self.settings.db_url)

    def create_embedder(self):
        if self.settings.embedding_provider == "sentence-transformers":
            return SentenceTransformerEmbedding(
                model_name=self.settings.embedding_model,
                device=self.settings.device,
            )
        elif self.settings.embedding_provider == "bert":
            return BERTEmbedder(
                model_name=self.settings.embedding_model,
                device=self.settings.device,
            )
        else:
            raise NotImplementedError(f"Provider '{self.settings.embedding_provider}' not supported")

    def get_embedding_dim(self, embedder) -> int:
        test_emb = embedder.embed_text("test")
        return test_emb.shape[0] if hasattr(test_emb, 'shape') else len(test_emb)

    def create_orm_model(self, embedding_dim: int):
        return create_embedding_model(dim=embedding_dim, table_name=self.settings.load_table_name)

    def create_splitter(self):
        embedder = self.create_embedder()
        sentence_splitter = SentenceSplitter()
        return SemanticChunker(
            embedder=embedder,
            threshold=0.35,
            min_chunk_size=1,
            sentence_splitter=sentence_splitter
        )

    def create_metadata_builder(self):
        field_mapping = {col: col for col in self.settings.metadata_columns} if self.settings.metadata_columns else {}
        return MetadataBuilder(field_mapping=field_mapping)

    def create_extractor(self, connector):
        return AsyncSQLExtractor(connector)

    def create_loader(self, *, orm_class):
        return AsyncSQLLoader(
            orm_class=orm_class,
            batch_size=self.settings.batch_size,
        )
    
    def create_transformer(
        self,
        embedder: SentenceTransformerEmbedding | BERTEmbedder,
        splitter: SentenceSplitter,
        metadata_builder: MetadataBuilder,
    ) -> Transformer:
        return Transformer(
            embedding=embedder,
            splitter=splitter,
            metadata_builder=metadata_builder,
            metadata_columns=self.settings.metadata_columns,
        )