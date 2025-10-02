from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .base import BaseConnector 


class SQLConnector(BaseConnector):
    def __init__(self, db_url:str):
        self.engine = create_engine(db_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
    
    def connect(self) -> Session:
        return self.SessionLocal()
    
    def close(self):
        return self.engine.dispose()
    
    def __enter__(self):
        self.session = self.connect()
        return self.session
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
    

