def ensure_normalized_similarity(similarity: float, backend_type) -> float:
    if similarity > 1.0 or similarity < 0.0:
        if 0 <= similarity <= 2:
            return max(0.0, 1.0 - (similarity / 2.0))
        else:
            raise ValueError(f"Unexpected similarity/distance value: {similarity}")
    return float(similarity)