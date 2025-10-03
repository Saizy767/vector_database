import os
import pytest

from sqlalchemy import Column, Integer, String, text
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

from vectordb.etl.loaders.sql_loader import SQLLoader
from vectordb.connector.sql_connector import SQLConnector

load_dotenv()

DB_URL = os.getenv("TEST_DB_URL")
if not DB_URL:
    raise RuntimeError("TEST_DB_URL не установлена в окружении")

Base = declarative_base()

class Book(Base):
    __tablename__ = 'table_book'
    __table_args__ = {"schema": "public"}

    table_name = 'table_book'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String) 


@pytest.fixture(scope='module')
def setup_db():
    connector = SQLConnector(DB_URL)

    Base.metadata.create_all(connector.engine)
    yield connector
    Base.metadata.drop_all(connector.engine)
    connector.close()


def test_bulk_insert(setup_db: SQLConnector):
    loader = SQLLoader(connector = setup_db,
                       table_name = Book.table_name,
                       orm_class=Book)
    data = [
        {"id": 1, "title": "Book A", "author": "Author 1"},
        {"id": 2, "title": "Book B", "author": "Author 2"}
    ]
    loader.load(data)

    with setup_db.connect() as session:
        result = session.execute(
            text(f"SELECT * FROM {Book.table_name} ORDER BY id")
            ).mappings().all()
        assert len(result) == 2
        assert result[0].title == "Book A"
        assert result[1].author == "Author 2"


def test_upsert(setup_db: SQLConnector):
    loader = SQLLoader(
        connector=setup_db,
        table_name=Book.table_name,
        conflict_update=['title', 'author'],
        conflict_target=['id']
    )

    # Вставляем существующую книгу с новым названием
    data = [{"id": 1, "title": "Updated Book A", "author": "Updated Author 1"}]
    loader.load(data)
    
    with setup_db.connect() as session:
        result = session.execute(text(f"SELECT * FROM {Book.table_name} WHERE id=1")).first()
        assert result.title == "Updated Book A"
        assert result.author == "Updated Author 1"


def test_empty_data(setup_db: SQLConnector):
    loader = SQLLoader(connector=setup_db, table_name=Book.table_name, orm_class=Book)
    loader.load([])
    # Проверяем, что ничего не сломалось
    with setup_db.connect() as session:
        result = session.execute(text(f"SELECT COUNT(*) FROM {Book.table_name}")).scalar()
        assert result >= 0


def test_batch_insert(setup_db: SQLConnector):
    loader = SQLLoader(connector=setup_db, table_name=Book.table_name, orm_class=Book, batch_size=1)
    data = [
        {"id": 3, "title": "Book C", "author": "Author 3"},
        {"id": 4, "title": "Book D", "author": "Author 4"}
    ]
    loader.load(data)

    with setup_db.connect() as session:
        result = session.execute(
            text(f"SELECT * FROM {Book.table_name} WHERE id IN (3,4)")).all()
        assert len(result) == 2

    

