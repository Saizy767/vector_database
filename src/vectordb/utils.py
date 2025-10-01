import numpy as np


def cosine_similarity(vector_a:np.ndarray, vector_b:np.ndarray):
    """Compute cosine similarity between query and a list of vectors"""
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )