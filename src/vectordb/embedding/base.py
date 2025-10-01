from typing import List

class BaseEmbedding:
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError
