import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .base import BaseConnector 

logger = logging.getLogger(__name__)

class SQLConnector(BaseConnector):
    def __init__(self, db_url:str):
        logger.info(f"Connecting to database (URL masked)")
        self.engine = create_engine(db_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
    
    def connect(self) -> Session:
        logger.debug("Opening new DB session")
        return self.SessionLocal()
    
    def close(self):
        logger.debug("Disposing DB engine")
        return self.engine.dispose()
    
    def __enter__(self):
        self.session = self.connect()
        return self.session
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
        logger.debug("DB session closed")
    

