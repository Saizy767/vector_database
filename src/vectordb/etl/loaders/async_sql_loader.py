import logging
from typing import List, Dict, Any, Type, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.vectordb.utils import batched

logger = logging.getLogger(__name__)


class AsyncSQLLoader:
    def __init__(
        self,
        orm_class: Type,
        batch_size: int = 100,
        conflict_update: Optional[List[str]] = None,
        conflict_target: Optional[List[str]] = None,
    ):
        if orm_class is None:
            raise ValueError("orm_class is required")
        self.orm_class = orm_class
        self.table = orm_class.__table__
        self.batch_size = batch_size
        self.conflict_update = conflict_update
        self.conflict_target = conflict_target or ["id"]
        logger.debug(f"AsyncSQLLoader initialized for table '{self.table.name}'")

    async def load(self, session: AsyncSession, data: List[Dict[str, Any]]):
        if not data:
            logger.debug("No data to load")
            return

        logger.info(f"Loading {len(data)} records into '{self.table.name}'")

        try:
            for batch in batched(data, self.batch_size):
                if self.conflict_update:
                    await self._upsert(session, batch)
                else:
                    await self._bulk_insert(session, batch)
            logger.debug("Async load completed")
        except Exception as e:
            logger.error(f"Error during async load: {e}")
            raise

    async def _bulk_insert(self, session: AsyncSession, data: List[Dict[str, Any]]):
        objects = [self.orm_class(**item) for item in data]
        session.add_all(objects)

    async def _upsert(self, session: AsyncSession, data: List[Dict[str, Any]]):
        column_map = {c.key: c.name for c in self.table.columns}
        mapped_data = []
        for item in data:
            mapped_item = {}
            for k, v in item.items():
                db_col_name = column_map.get(k, k)
                mapped_item[db_col_name] = v
            mapped_data.append(mapped_item)

        dialect = session.bind.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(self.table).values(mapped_data)
            update_cols = [column_map.get(col, col) for col in self.conflict_update]
            update_dict = {
                c.name: stmt.excluded[c.name]
                for c in self.table.columns
                if c.name in update_cols
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=self.conflict_target,
                set_=update_dict,
            )
            await session.execute(stmt)
        else:
            logger.warning(f"Upsert not supported for dialect '{dialect}', using INSERT")
            await self._bulk_insert(session, data)