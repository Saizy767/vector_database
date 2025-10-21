from .settings import settings
from search.backend.pgvector_backend import PGVectorBackend

def create_search_backend():
    backend_type = settings.search_backend
    if backend_type == "pgvector":
        config = settings.get_backend_config()
        return PGVectorBackend(db_url=config["db_url"])
    else:
        raise ValueError(f"Unknown backend: {backend_type}")