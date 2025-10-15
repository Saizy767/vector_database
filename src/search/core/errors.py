class SearchError(Exception):
    pass

class BackendConnectionError(SearchError):
    pass

class EmbeddingError(SearchError):
    pass

class InvalidQueryError(SearchError):
    pass