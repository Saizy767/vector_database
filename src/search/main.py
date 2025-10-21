import logging
import uvicorn
from search.api.app import app

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)