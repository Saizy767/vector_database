import logging
import asyncio
from etl.pipeline import ETLPipeline
from etl.config import ETLSettings

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def main():
    settings = ETLSettings()
    pipeline = ETLPipeline(settings)
    asyncio.run(pipeline.run()) 

if __name__ == "__main__":
    main()