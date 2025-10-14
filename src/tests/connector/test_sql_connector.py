import os
import pytest
from sqlalchemy import text
from etl.core.connector.sql_connector import SQLConnector
from dotenv import load_dotenv


load_dotenv()
DB_URL = os.getenv("TEST_DB_URL")

if not DB_URL:
    raise RuntimeError("TEST_DB_URL не установлена в окружении")

@pytest.fixture(scope="module")
def connector():
    _connector = SQLConnector(DB_URL)
    yield _connector
    _connector.close()

def test_connector_connect_and_query(connector: SQLConnector):
    with connector.connect() as session:
        query = text('SELECT 1')
        result = session.execute(query).scalar()
        assert result == 1

def test_connector_create_and_insert(connector: SQLConnector):
    table_name = "public.test_connector_table"
    with connector.connect() as session:
        session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        session.execute(text(f"CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, name TEXT)"))
        session.execute(text(f"INSERT INTO {table_name} (name) VALUES ('Alice')"))
        session.commit()

        result = session.execute(text(f"SELECT name FROM {table_name} WHERE id=1")).scalar()
        assert result == "Alice"
        session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))