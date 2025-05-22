import os

from dotenv import load_dotenv

load_dotenv()

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Path configuration
PROJECT_BASE_PATH = os.getenv("PROJECT_BASE_PATH")
if PROJECT_BASE_PATH is None:
    raise ValueError("PROJECT_BASE_PATH environment variable is not set.")
CLEAN_DATA_FILE = os.path.join(
    PROJECT_BASE_PATH, "output", "data", "clean_data.parquet"
)
FILTERED_DATA_FILE = os.path.join(
    PROJECT_BASE_PATH, "output", "data", "region_filtered_data.parquet"
)
PROBES_FOLDER = os.path.join(PROJECT_BASE_PATH, "input", "probes_data")
GRAPH_FILE = os.path.join(
    PROJECT_BASE_PATH, "output", "graphs", "portugal_full_graph.graphml"
)
MAPPED_DATA_FILE = os.path.join(
    PROJECT_BASE_PATH, "output", "data", "segment_mapped_data.parquet"
)
EDGES_DATA_FILE = os.path.join(PROJECT_BASE_PATH, "output", "edges", "edges_data.pkl")
FEATURES_DATA_FILE = os.path.join(
    PROJECT_BASE_PATH, "output", "data", "features_data.parquet"
)

# Model paths
MODEL_PATH = os.path.join(PROJECT_BASE_PATH, "output", "models", "model.pkl")
SCALER_PARAMS_PATH = os.path.join(
    PROJECT_BASE_PATH, "output", "models", "scaler_params.pkl"
)
LABEL_ENCODER_PATH = os.path.join(
    PROJECT_BASE_PATH, "output", "models", "label_encoder.pkl"
)

# Region configuration
REGION_TYPE = "city"
REGION_NAME = "Gaia"

# Filters Parser
TS_START = "2020-08-01"
TS_END = "2020-11-30"
MAX_HDOP = 10.0
MAX_SPEED = 120.0

# Filters Features
OFF_PEAK_HOURS = [1, 2, 3, 4, 5, 9, 10, 11, 15, 16, 21, 22, 23, 24]
