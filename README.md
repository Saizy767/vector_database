# VectorDB ETL Pipeline

## 📘 Overview

The **VectorDB ETL Pipeline** is a modular and extensible system designed to extract text data from a relational database, transform it into **vector embeddings** (using transformer models such as BERT or Sentence Transformers), and load the processed data back into another SQL table.

This project is ideal for applications involving **semantic search**, **vector databases**, or **AI-powered document retrieval**.

---

## 🧩 Project Structure

```
project/
│
├── main.py                     # Entry point: runs the full ETL pipeline
├── src/
│   ├── settings.py             # Pydantic-based configuration loader (.env support)
│   ├── models/                 # SQLAlchemy ORM models
│   ├── vectordb/               # Core logic for Vector ETL
│   │   ├── connector/          # Database connectors (SQLConnector)
│   │   ├── etl/                # Extractors, Transformers, Loaders
│   │   ├── embedding/          # Embedding models (BERT, Sentence Transformers)
│   │   ├── metadata/           # Metadata builder for chunks
│   │   ├── splitters/          # Sentence splitting utilities
│   │   └── utils.py            # Helper functions (logging, cosine similarity)
│   └── tests/                  # Comprehensive pytest-based test suite
│
└── .env                        # Environment variables (not tracked by Git)
```

---

## ⚙️ How It Works

1. **Extraction**  
   - The pipeline extracts data from a source SQL table using `SQLExtractor`.  
   - Data is processed in configurable batches for performance.

2. **Transformation**  
   - Each text entry is split into sentences or smaller chunks using `SentenceSpliter`.
   - Chunks are embedded using `BERTEmbedder` or `SentenceTransformerEmbedding`.
   - Metadata for each chunk is built and validated via `MetadataBuilder`.

3. **Loading**  
   - Processed embeddings and metadata are bulk inserted (or upserted) into a target SQL table using `SQLLoader`.

---

## 🧾 Example `.env` Configuration

```bash
DB_URL=postgresql+psycopg2://user:password@localhost:5432/mydb
EXTRACT_TABLE_NAME=source_table
LOAD_TABLE_NAME=embedding_chapter
SOURCE_ID=id
EMBEDDING_COLUMNS=chapter_text
METADATA_COLUMNS=chapter_title,chapter_number
EMBEDDING_PROVIDER=bert
EMBEDDING_MODEL=bert-base-uncased
DEVICE=cuda
```

---

## ▶️ Getting Started

### 1️⃣ Setup

```bash
git clone <repository-url>
cd project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

Edit `.env` and specify all required parameters as shown above.

### 3️⃣ Run the Pipeline

```bash
python main.py
```

Logs will be saved to `app.log` and printed to the console.

---

## 🧪 Testing

To run the full test suite:

```bash
export PYTHONPATH=$(pwd)/src
pytest src/tests/embedding/ -v --cov=vectordb.embedding
pytest src/tests/connector/ -v --cov=vectordb.connector
pytest src/tests/etl/extractors/ -v --cov=vectordb.etl.extractors
pytest src/tests/etl/loaders/ -v --cov=vectordb.etl.loaders
pytest src/tests/splitters/ -v --cov=vectordb.splitters
pytest src/tests/metadata/ -v --cov=vectordb.metadata
```

---

## 🧠 Key Components

| Component | Description |
|------------|-------------|
| **SQLConnector** | Creates and manages SQLAlchemy sessions |
| **SQLExtractor** | Fetches data in batches from SQL tables |
| **Transformer** | Coordinates splitting, embedding, and metadata creation |
| **SQLLoader** | Loads (inserts/upserts) transformed data into SQL |
| **SentenceSpliter** | Splits long texts into sentences/chunks |
| **MetadataBuilder** | Generates structured metadata for each text chunk |
| **EmbeddingChapter** | ORM model for embedding storage |

---

## 🛠️ Extensibility

- Add new embedding models by subclassing `BaseEmbedding`.
- Add new extractors/loaders by extending `BaseExtractor` or `BaseLoader`.
- Customize metadata structure through `.env` configuration or `MetadataBuilder` overrides.

---

## 🧑‍💻 Development Notes

- Python 3.10+ required
- GPU acceleration supported (via `cuda` or `mps`)
- Compatible with PostgreSQL, MySQL, and SQLite

---
