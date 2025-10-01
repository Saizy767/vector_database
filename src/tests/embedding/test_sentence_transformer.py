import pytest
import numpy as np

from vectordb.embedding.sentence_transformer import SentenсeTransformerEmbedding

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEVICE = 'mps'


@pytest.fixture
def embedder() -> SentenсeTransformerEmbedding:
    return SentenсeTransformerEmbedding(model_name=MODEL_NAME, device=DEVICE)

def test_single_emb(embedder: SentenсeTransformerEmbedding):
    text = 'Проверка текста'
    embedding = embedder.embed_text(text)

    assert isinstance(embedding, np.ndarray), "Эмбеддинг должен быть numpy.ndarray"
    assert embedding.ndim == 1, "Эмбеддинг должен быть 1-мерным"
    assert embedding.shape[0] > 0, "Эмбеддинг не должен быть пустым"


def test_similarity_embeddings(embedder: SentenсeTransformerEmbedding):
    text = "Проверка на схожесть"
    emb1 = embedder.embed_text(text)
    emb2 = embedder.embed_text(text)

    np.testing.assert_allclose(emb1, emb2, rtol=1e-5)
