import json
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

class UTF8JSON(TypeDecorator):
    impl = JSONB
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value, ensure_ascii=False)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value