import logging
from .abc import BaseSearchBackend, BaseEmbeddingProvider
from .query import SearchQuery, SearchResponse
from .similarity import ensure_normalized_similarity

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(
        self,
        backend: BaseSearchBackend,
        embedder: BaseEmbeddingProvider,
        cache=None
    ):
        self.backend = backend
        self.embedder = embedder
        self.cache = cache

    async def search(self, query: SearchQuery) -> SearchResponse:
        vector = self.embedder.embed_text(query.text)

        if self.cache:
            cache_key = self._make_cache_key(vector, query)
            cached = await self.cache.get(cache_key)
            if cached:
                return SearchResponse.model_validate(cached)

        raw_results = await self.backend.search(
            query_vector=vector,
            top_k=query.top_k,
            min_similarity=query.min_similarity,
            metadata_filter=query.metadata_filter
        )

        normalized_results = [
            {**r, "similarity": ensure_normalized_similarity(r["similarity"], backend_type=type(self.backend))}
            for r in raw_results
        ]

        response = SearchResponse(
            results=normalized_results,
            total=len(normalized_results)
        )

        if self.cache:
            await self.cache.set(cache_key, response.model_dump(), ttl=300)

        return response

    def _make_cache_key(self, vector: list, query: SearchQuery) -> str:
        import hashlib
        key_str = f"{str(vector)}|{query.model_dump_json()}"
        return hashlib.md5(key_str.encode()).hexdigest()