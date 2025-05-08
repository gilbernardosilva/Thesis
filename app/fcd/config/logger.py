import logging
import os

from app.fcd.config.variables import BASE_PATH, LOG_FORMAT, LOG_LEVEL

log_file = os.path.join(BASE_PATH, "output", "pipeline.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)

logging.debug("Logger initialized.")
