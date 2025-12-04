from .settings import settings
from search.backend.pgvector_backend import PGVectorBackend
from search.backend.faiss_backend import FAISSBackend

def create_search_backend():
    backend_type = settings.search_backend
    if backend_type == "pgvector":
        return PGVectorBackend(
            db_url=settings.db_url,
            table_name=settings.load_table_name
        )
    elif backend_type == "faiss":
        return FAISSBackend(
            index_path=settings.faiss_index_path,
            metadata_path=settings.faiss_metadata_path
        )
    else:
        raise ValueError(f"Unknown backend: {backend_type}")