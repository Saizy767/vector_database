import logging

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

import asyncio
from etl.pipeline import ETLPipeline
from etl.config import ETLSettings

def main():
    settings = ETLSettings()
    pipeline = ETLPipeline(settings)
    if settings.async_mode:
        print('START ASYNC')
        asyncio.run(pipeline.run_async())
    else:
        pipeline.run_sync()

if __name__ == "__main__":
    main()