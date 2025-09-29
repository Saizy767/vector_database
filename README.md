This repository contains a universal vector database implementation in Python.

## Features
- Separate storage for embeddings and metadata.
- Memory and SQL backends.
- Brute-force vector search (easily replaceable with FAISS/HNSW).
- Simple API for upsert, search, update, and delete.
- Dummy embedding provider for testing.

## Project Structure
```
src/vectordb/
├── core.py               # Main VectorDB class
├── models.py             # DocumentMetadata dataclass
├── utils.py              # Vector utilities (normalization, cosine similarity)
├── storage/
│   ├── interface.py      # Storage interface
│   ├── stub.py           # Stub in-memory storage for testing
│   ├── memory.py         # Memory backend
│   └── sql.py            # SQL backend
├── embeddings/
│   ├── base.py           # Embedding interface
│   └── dummy.py          # Dummy embedding
├── index/
│   └── brute_force.py    # Brute-force search index
└── examples/
    ├── example_usage.py  # Example usage
    └── example_stub_usage.py # Example stub usage
tests/
└── test_vectordb.py      # Unit tests for VectorDB

```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Example
```bash
export PYTHONPATH=$(pwd)/src
python3 src/vectordb/examples/example_usage.py
python3 src/vectordb/examples/example_stub_usage.py
```

## Running Tests
```bash
export $PYTHONPATH:$(pwd)/src
pytest src/tests/ -v
pytest --cov=vectordb src/tests/ -v
```
