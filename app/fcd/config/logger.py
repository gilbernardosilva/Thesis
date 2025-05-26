import logging
import os

import pandas as pd

from app.fcd.config.variables import LOG_FORMAT, LOG_LEVEL, PROJECT_BASE_PATH

log_file = os.path.join(PROJECT_BASE_PATH, "output", "pipeline.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)

logging.debug("Logger initialized.")
pd.set_option("future.no_silent_downcasting", True)
logging.getLogger("distributed").setLevel(logging.ERROR)
