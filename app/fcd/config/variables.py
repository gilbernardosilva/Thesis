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

# Region to be filtered configuration
REGION_TYPE = "city"
REGION_NAME = "Gaia"

# Filters for the data processing script
TS_START = "2020-08-01"
TS_END = "2020-11-30"
MAX_HDOP = 10.0
MAX_SPEED = 120.0

# Filters Features
PEAK_HOURS = [7, 8, 12, 13, 14, 17, 18, 19]  # 7–9 AM, 12–3 PM, 5–8 PM
OFF_PEAK_HOURS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    10,
    11,
    15,
    16,
    20,
    21,
    22,
    23,
]  # 0–6 AM, 9–11 AM, 3–4 PM, 8–11 PM

DASK_MEMORY_LIMIT = "3GB"
DASK_NUM_THREADS = 2
DASK_NUM_CORES = 2

# SUMO configuration
SUMO_HOME = os.getenv("SUMO_HOME")
SUMO_PROJECT_PATH = os.path.join(PROJECT_BASE_PATH, "sumo")
SUMO_CONFIG_FILE = os.path.join(SUMO_PROJECT_PATH, "input", "gaia.sumocfg")
SUMO_NET_FILE = os.path.join(SUMO_PROJECT_PATH, "input", "gaia.net.xml.gz")
SUMO_TRIPS_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "gaia_new_trips.xml")
SUMO_FCD_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "gaia_fcd.xml")
SUMO_ACCIDENTS_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "accidents.parquet")
SUMO_ROUTES_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "gaia_routes.rou.xml")
SUMO_FCD_PARSED_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "parsed_fcd.parquet")
SUMO_MATCHED_FILE = os.path.join(SUMO_PROJECT_PATH, "output", "matched_fcd.parquet")

SUMO_END_TIME = 3600  # Seconds (60  *  60)
SUMO_ACCIDENT_PROB = 0.01
SUMO_MIN_ACCIDENT_DURATION = 300  # Seconds
SUMO_MAX_ACCIDENT_DURATION = 6000  # Seconds
SUMO_STARTING_HOUR = 7

SUMO_ACCIDENTS_PREDICTION_FILE = os.path.join(
    SUMO_PROJECT_PATH, "output", "accidents_prediction.csv"
)
