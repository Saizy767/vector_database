import os
import pytest
from dotenv import load_dotenv
from vectordb.storage.sql import SQLStorage
from vectordb.core import VectorDB
from vectordb.embeddings.dummy import DummyEmbedding

# Загружаем DB_TEST_URL из .env
load_dotenv()
DB_TEST_URL = os.getenv("DB_URL")
if not DB_TEST_URL:
    raise RuntimeError("DB_TEST_URL is not set in .env (add a test DB!)")


@pytest.fixture(scope="module")
def vector_db():
    """Фикстура для VectorDB с SQLStorage и DummyEmbedding."""
    storage = SQLStorage(DB_TEST_URL)
    embedder = DummyEmbedding()
    db = VectorDB(storage=storage, embedder=embedder)
    yield db

    # Очистка всех тестовых документов после завершения тестов
    for doc_id in ["test_doc_1", "test_doc_2", "test_doc_3"]:
        try:
            db.delete(doc_id, hard=True)
        except Exception:
            pass


@pytest.fixture
def test_doc():
    return {
        "doc_id": "test_doc_1",
        "text": "Hello, this is a test document.",
        "metadata": {"category": "test"}
    }


@pytest.fixture(autouse=True)
def cleanup_before_test(vector_db, test_doc):
    """Удаляем документ перед каждым тестом, чтобы тесты были изолированы."""
    try:
        vector_db.delete(test_doc["doc_id"], hard=True)
    except Exception:
        pass
    yield
    

def test_upsert_and_search(vector_db, test_doc):
    """Проверяем вставку документа и поиск."""
    doc_id = test_doc["doc_id"]
    text = test_doc["text"]
    metadata = test_doc["metadata"]
    embedding = vector_db.embedder.embed_text(text)

    # Вставляем документ
    vector_db.upsert(doc_id=doc_id, embedding=embedding, metadata=metadata, text=text)

    # Проверяем, что документ вернулся через поиск
    results = vector_db.search(query_embedding=embedding, k=1)
    assert len(results) == 1
    meta, score = results[0]
    assert meta.id == doc_id
    assert meta.metadata["category"] == "test"


def test_update_metadata(vector_db):
    doc_id = "test_doc_1"
    text = "Hello, this is a test document."
    metadata = {"category": "test"}
    embedding = vector_db.embedder.embed_text(text)
    
    # Вставляем документ
    vector_db.upsert(doc_id=doc_id, embedding=embedding, metadata=metadata, text=text)

    # Проверяем, что документ есть в БД
    doc = vector_db.storage.get_metadata(doc_id=doc_id)
    assert doc.id == doc_id, "Документ не был вставлен в БД"

    # Обновляем метаданные
    new_metadata = {"category": "updated"}
    vector_db.update_metadata(doc_id, new_metadata)

    # Поиск по embedding вставленного текста
    results = vector_db.search(query_embedding=embedding, k=1)

    assert len(results) > 0, "Поиск не вернул результатов"
    meta, score = results[0]
    assert meta.metadata["category"] == "updated"


def test_soft_delete(vector_db, test_doc):
    """Проверяем мягкое удаление документа."""
    doc_id = test_doc["doc_id"]

    # Soft delete документа
    vector_db.delete(doc_id)
    
    results = vector_db.search(query_embedding=vector_db.embedder.embed_text("Hello"), k=1)
    # Документ должен быть помечен как удаленный и не возвращаться
    assert len(results) == 0


def test_hard_delete(vector_db):
    """Проверяем физическое удаление документа."""
    doc_id = "test_doc_2"
    text = "Temporary doc for hard delete test"
    embedding = vector_db.embedder.embed_text(text)
    metadata = {"category": "temp"}

    # Вставляем документ
    vector_db.upsert(doc_id=doc_id, embedding=embedding, metadata=metadata, text=text)
    
    # Hard delete
    vector_db.delete(doc_id, hard=True)

    results = vector_db.search(query_embedding=embedding, k=1)
    assert len(results) == 0
