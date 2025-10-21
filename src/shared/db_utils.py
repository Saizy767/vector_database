from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

async def get_embedding_dim_from_db(engine: AsyncEngine, table_name: str) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(f"""
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = '{table_name}'::regclass
                  AND attname = 'embedding';
            """)
        )
        row = result.fetchone()
        if not row:
            raise RuntimeError(f"Таблица '{table_name}' не содержит колонки 'embedding'")
        # В pgvector typmod = dim + 4
        typmod = row[0]
        if typmod < 0:
            raise RuntimeError(f"Невозможно определить размерность из typmod={typmod}")
        return typmod