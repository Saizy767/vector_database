from .settings import settings
from search.backend.pgvector_backend import PGVectorBackend

def create_search_backend():
    backend_type = settings.search_backend
    backend_config = settings.get_backend_config()

    if backend_type == "pgvector":
        return PGVectorBackend(db_url=backend_config.db_url)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")