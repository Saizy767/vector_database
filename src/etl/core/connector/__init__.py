"""
Модуль коннекторов для баз данных в VectorDB ETL пайплайне.

Предоставляет абстракции и реализации для подключения к различным источникам данных:
- sql_connector.py: Синхронный коннектор для PostgreSQL
- async_sql_connector.py: Асинхронный коннектор для PostgreSQL
- base.py: Абстрактный базовый класс для всех коннекторов

Особенности реализации:
- Поддержка синхронного и асинхронного режимов работы
- Управление пулом соединений
- Автоматическое закрытие соединений (context manager)
- Поддержка транзакций
- Логирование всех операций с БД

Использование:

from etl.core.connector import SQLConnector

# Синхронный режим
connector = SQLConnector("postgresql://user:pass@localhost/db")
with connector.connect() as session:
    result = session.execute(text("SELECT * FROM table"))
    rows = result.fetchall()

# Асинхронный режим
connector = AsyncSQLConnector("postgresql://user:pass@localhost/db")
session = await connector.connect()

Этот модуль обеспечивает изоляцию бизнес-логики от деталей подключения к БД
и позволяет легко переключаться между различными СУБД и режимами работы.
"""

from .sql_connector import SQLConnector

__all__ = ["SQLConnector"]
