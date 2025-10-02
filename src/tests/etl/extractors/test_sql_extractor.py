import os
import pytest
from dotenv import load_dotenv
from sqlalchemy import text

from vectordb.connector.sql_connector import SQLConnector
from vectordb.etl.extractors.sql_extractor import SQLExtractor

load_dotenv()

table_name = "public.book_test"
DB_URL = os.getenv("TEST_DB_URL")
if not DB_URL:
    raise RuntimeError("TEST_DB_URL не установлена в окружении")

@pytest.fixture(scope="module")
def setup_db():
    connector = SQLConnector(DB_URL)

    with connector.connect() as session:
        session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        session.execute(text(f"""
            CREATE TABLE {table_name} (
                book_id SERIAL PRIMARY KEY,
                chapter_number INT,
                chapter_title TEXT,
                chapter_text TEXT
            )
        """))
        session.execute(text(f"""
            INSERT INTO {table_name} (chapter_number, chapter_title, chapter_text)
            VALUES
                (1, 'Intro', 'This is the first chapter'),
                (2, 'Middle', 'This is the second chapter'),
                (3, 'Finale', 'This is the last chapter')
        """))
        session.commit()
    yield connector

    with connector.connect() as session:
        session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        session.commit()
    connector.close()

@pytest.fixture(scope='module')
def extractor(setup_db):
    return SQLExtractor(setup_db)


def test_extract_all(extractor: SQLExtractor):
    rows = extractor.extract_all(table_name)
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"book_id", "chapter_number", "chapter_title", "chapter_text"}
    
def test_extract_batch(extractor: SQLExtractor):
    batches = list(extractor.extract_batches(table_name, batch_size=2))
    assert len(batches) == 2
    assert [row["book_id"] for row in batches[0]] == [1, 2]
    assert [row["book_id"] for row in batches[1]] == [3]
    
def test_extract_all_spec_columns(extractor: SQLExtractor):
    rows = extractor.extract_all(table_name, columns=["book_id", "chapter_title"])
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"book_id", "chapter_title"}

    
def test_extract_batch_spec_columns(extractor: SQLExtractor):
    batches = list(extractor.extract_batches(table_name, columns=["book_id"], batch_size=2))
    assert len(batches) == 2
    assert [row["book_id"] for row in batches[0]] == [1, 2]
    assert [row["book_id"] for row in batches[1]] == [3]

def test_extract_all_empty_table(setup_db):
    connector = setup_db
    with connector.connect() as session:
        session.execute(text(f"DELETE FROM {table_name}"))
        session.commit()

    extractor = SQLExtractor(connector)
    rows = extractor.extract_all(table_name)
    assert rows == []


def test_extract_batches_empty_table(setup_db):
    extractor = SQLExtractor(setup_db)
    batches = list(extractor.extract_batches(table_name, batch_size=2))
    assert batches == []
