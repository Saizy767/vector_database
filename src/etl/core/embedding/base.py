import numpy as np

class BaseEmbedding:
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError
