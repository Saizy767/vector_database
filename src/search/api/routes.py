import logging
from fastapi import APIRouter, HTTPException
from search.core.query import SearchQuery, SearchResponse
import search.api.state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(query: SearchQuery):
    service = search.api.state.search_service
    if service is None:
        logger.error("Search service not ready")
        raise HTTPException(status_code=503, detail="Search service is initializing")

    try:
        response = await service.search(query)
        return response
    except Exception as e:
        logger.exception("Search failed due to internal error")
        raise HTTPException(status_code=500, detail="Search service unavailable")