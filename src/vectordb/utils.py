import numpy as np
import logging


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        return float(np.dot(a, b) / denom)
    if a.ndim == 1 and b.ndim == 2:
        b_norms = np.linalg.norm(b, axis=1) + eps
        return np.dot(b, a) / (np.linalg.norm(a) * b_norms)
    if a.ndim == 2 and b.ndim == 2:
        a_norms = np.linalg.norm(a, axis=1)[:, None] + eps
        b_norms = np.linalg.norm(b, axis=1)[None, :] + eps
        return (a @ b.T) / (a_norms * b_norms)
    raise ValueError("Unsupported shapes")


def get_logger(name="vectordb"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger