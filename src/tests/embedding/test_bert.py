import numpy as np
import pytest

from vectordb.embedding.bert import BERTEmbedder
from vectordb.utils import cosine_similarity


MODEL_NAME = 'bert-base-uncased'
DEVICE = 'mps'

@pytest.fixture(scope="module")
def bert_embedder_norm():
    # создаём один экземпляр для всех тестов
    return BERTEmbedder(model_name=MODEL_NAME, normalize=True, device=DEVICE)

@pytest.fixture(scope="module")
def bert_embedder_raw():
    return BERTEmbedder(model_name=MODEL_NAME, normalize=False, device=DEVICE)


def test_single_text_embedding_normalized(bert_embedder_norm:BERTEmbedder):
    text = "Hello world! This is a BERT embedding test."

    embedding = bert_embedder_norm.embed_text(text)

    assert isinstance(embedding, np.ndarray), "Эмбеддинг должен быть numpy.ndarray"
    assert embedding.shape == (768,), f"Размерность эмбеддинга должна быть 768, а не {embedding.shape}"
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f"Эмбеддинг должен быть нормализован, а норма = {norm}"


def test_single_text_embedding_raw(bert_embedder_raw:BERTEmbedder):
    text = "Just another test for raw embedding."
    embedding = bert_embedder_raw.embed_text(text)

    # Тип и размерность
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

    # Норма должна быть > 1
    norm = np.linalg.norm(embedding)
    assert norm > 1.0, f"Эмбеддинг без нормализации должен иметь норму > 1, а не {norm}"


def test_similarity_of_texts(bert_embedder_norm):
    text1 = "The cat is sitting on the mat."
    text2 = "A kitten is resting on the rug."
    text3 = "The stock market crashed yesterday."

    emb1 = bert_embedder_norm.embed_text(text1)
    emb2 = bert_embedder_norm.embed_text(text2)
    emb3 = bert_embedder_norm.embed_text(text3)

    sim_close = cosine_similarity(emb1, emb2)
    sim_far = cosine_similarity(emb1, emb3)

    assert sim_close > sim_far, (
        f"Схожие тексты должны иметь большую близость ({sim_close:.3f}) "
        f"чем несхожие ({sim_far:.3f})"
    )
