from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from .base import BaseConnector 

class AsyncSQLConnector(BaseConnector):
    def __init__(self, db_url: str):
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        elif not db_url.startswith("postgresql+asyncpg://"):
            raise ValueError("Only PostgreSQL with asyncpg is supported in AsyncSQLConnector")
        self.engine = create_async_engine(db_url, echo=False)
        self.SessionLocal = async_sessionmaker(self.engine, expire_on_commit=False)

    async def connect(self) -> AsyncSession:
        return self.SessionLocal()

    async def close(self):
        await self.engine.dispose()