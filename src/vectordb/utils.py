import hashlib
import json
import numpy as np
from typing import List


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length"""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def cosine_similarity_matrix(query: List[float], vectors: List[List[float]]):
    """Compute cosine similarity between query and a list of vectors"""
    q = np.array(query, dtype=np.float32)
    v = np.array(vectors, dtype=np.float32)
    return v @ q / (np.linalg.norm(v, axis=1) * np.linalg.norm(q) + 1e-10)

def hash_embedding(vec: List[float], method: str = "sha256") -> str:
    """Generate a hash id from a normalized embedding"""
    norm_vec = normalize_vector(vec)
    vec_str = ",".join(f"{x:.6f}" for x in norm_vec)
    if method == "sha256":
        return hashlib.sha256(vec_str.encode()).hexdigest()
    elif method == "md5":
        return hashlib.md5(vec_str.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash method: {method}")

def serialize_embedding(vec: List[float]) -> str:
    """Serialize embedding to JSON string"""
    return json.dumps(vec)

def deserialize_embedding(data: str) -> List[float]:
    """Deserialize embedding from JSON string"""
    return json.loads(data)