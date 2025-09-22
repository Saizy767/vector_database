import numpy as np
from typing import Iterable, List


def normalize_vector(vec: Iterable[float]) -> List[float]:
    a = np.array(list(vec), dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0:
        return a.tolist()
    return (a / norm).tolist()


def cosine_similarity_matrix(query: Iterable[float], vectors: List[Iterable[float]]) -> List[float]:
    q = np.array(list(query), dtype=float)
    qnorm = np.linalg.norm(q)
    vs = np.array([list(v) for v in vectors], dtype=float)
    vnorms = np.linalg.norm(vs, axis=1)
    denom = (qnorm * vnorms)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        sims = (vs @ q) / denom
        sims = np.nan_to_num(sims)
    return sims.tolist()