import os
from dotenv import load_dotenv

from vectordb.storage.sql import SQLStorage
from vectordb.core import VectorDB
from vectordb.embeddings.dummy import DummyEmbedding

def demo_sql():
    # Load DB_URL from .env
    load_dotenv()
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is not set in .env")

    # Initialize storage + db
    storage = SQLStorage(db_url)
    embedder = DummyEmbedding()
    db = VectorDB(storage=storage, embedder=embedder)

    # Insert a document
    doc_id = "sql_doc_1"
    text = "Hello from SQL backend!"
    embedding = embedder.embed_text(text)
    metadata = {"category": "greeting"}
    db.upsert(doc_id=doc_id, embedding=embedding, metadata=metadata, text=text)
    print(f"Inserted document {doc_id}")

    # Search for similar document
    query = embedder.embed_text("Hello world")
    results = db.search(query_embedding=query, k=3)
    print("Search results:")
    for meta, score in results:
        print(f"- {meta.id}, score={score:.4f}, metadata={meta.metadata}")

    # Update metadata
    db.update_metadata(doc_id, {"category": "updated"})
    print("Updated metadata")

    # Soft delete
    db.delete(doc_id)
    print(f"Soft-deleted document {doc_id}")

if __name__ == "__main__":
    demo_sql()
