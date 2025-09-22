from vectordb.storage.stub import StubStorage
from vectordb.core import VectorDB
from vectordb.utils import normalize_vector

class DummyEmbedder:
    """Simple dummy embedder for testing"""
    def embed_text(self, text: str):
        # Return a small vector based on character codes
        return normalize_vector([ord(c) % 10 for c in text[:5]])

def demo_stub():
    storage = StubStorage()
    embedder = DummyEmbedder()
    db = VectorDB(storage=storage, embedder=embedder)

    # === Upsert with explicit id ===
    text = 'Hello world'
    emb = embedder.embed_text(text)
    db.upsert(doc_id='doc_stub_1', embedding=emb, metadata={'lang': 'en'}, text=text)

    # === Upsert with auto-generated id ===
    text2 = 'Testing stub'
    emb2 = embedder.embed_text(text2)
    db.upsert(embedding=emb2, metadata={'lang': 'en'}, text=text2)

    # === Get document ===
    db.get('doc_stub_1')

    # === Search ===
    db.search(query_embedding=embedder.embed_text('Hello'))

    # === Update metadata ===
    db.update_metadata('doc_stub_1', {'tag': 'test'})

    # === Soft delete ===
    db.delete('doc_stub_1')

if __name__ == '__main__':
    demo_stub()