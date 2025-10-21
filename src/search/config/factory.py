from .settings import settings
from search.backend.pgvector_backend import PGVectorBackend

def create_search_backend():
    backend_type = settings.search_backend
    if backend_type == "pgvector":
        return PGVectorBackend(
            db_url=settings.db_url,
            table_name=settings.load_table_name
        )
    else:
        raise ValueError(f"Unknown backend: {backend_type}")