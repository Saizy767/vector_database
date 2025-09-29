from vectordb.storage.memory import MemoryStorage
from vectordb.embeddings.dummy import DummyEmbedding
from vectordb.core import VectorDB



def demo():
    storage = MemoryStorage()
    embedder = DummyEmbedding(dim=32)
    db = VectorDB(storage=storage, embedder=embedder)
    docs = [
        ('doc1', embedder.embed_text('The quick brown fox'), {'source': 'a', 'lang': 'en'}, 'The quick brown fox'),
        ('doc2', embedder.embed_text('A fast brown animal'), {'source': 'b', 'lang': 'en'}, 'A fast brown animal'),
        ('doc3', embedder.embed_text('I love pizza and pasta'), {'source': 'c', 'lang': 'it'}, 'I love pizza and pasta'),
    ]

    for doc_id, emb, meta, text in docs:
        db.upsert(doc_id=doc_id, embedding=emb, metadata=meta, text=text)


    # === Update metadata ===
    print('Update metadata for doc1')
    db.update_metadata('doc1', {'tag': 'animal'})
    updated_meta, _ = db.get('doc1')
    print('Get doc1:', updated_meta)


    # === Soft delete ===
    print('Soft delete doc2')
    db.delete('doc2')
    print('Remaining ids with embeddings:', storage.all_ids_with_embeddings())


    # === Upsert new doc with auto-generated id ===
    new_text = 'A clever brown fox'
    new_emb = embedder.embed_text(new_text)
    db.upsert(embedding=new_emb, metadata={'source': 'd', 'lang': 'en'}, text=new_text)
    print('New document added with auto-generated id')


if __name__ == '__main__':
    demo()