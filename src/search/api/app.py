import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from search.api.routes import router as search_router
from search.api.state import search_service
from search.config.factory import create_search_backend
from search.config.settings import settings
from shared.embedding.sentence_transformer import SentenceTransformerEmbedding
from search.core.embedding import SharedEmbeddingAdapter
from search.core.service import SearchService

search_service: SearchService | None = None
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = create_search_backend()
    embedder_model = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        device=settings.device
    )
    embedder = SharedEmbeddingAdapter(embedder=embedder_model)
    service = SearchService(backend=backend, embedder=embedder)

    import search.api.state
    search.api.state.search_service = service
    logger.info("✅ SearchService initialized")
    yield
    await backend.disconnect()
    logger.info("🛑 SearchService shut down")

app = FastAPI(
    title="VectorDB Search API",
    description="API для семантического поиска по векторной БД",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(search_router)