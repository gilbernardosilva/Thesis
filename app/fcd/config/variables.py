import os

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Path configuration
BASE_PATH = "/Users/gilsilva/Work/thesis"
INPUT_CSV = os.path.join(BASE_PATH, "input", "probes", "probes-PRT.2025.02.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "output", "probes", "probes-prt-processed.csv")
CLEAN_DF = os.path.join(BASE_PATH, "output", "dataframes", "cleaned_df.parquet")
FILTERED_DF = os.path.join(BASE_PATH, "output", "dataframes", "filtered_df.parquet")
PROBES_FOLDER = os.path.join(BASE_PATH, "input", "probes")
OSM_FILE = os.path.join(BASE_PATH, "input", "portugal_roads.osm")
GRAPH_FILE = os.path.join(BASE_PATH, "output", "graphs", "portugal_full_graph.graphml")
MAPPED_DF = os.path.join(BASE_PATH, "output", "dataframes", "mapped_df.parquet")
EDGES_DATA_FILE = os.path.join(BASE_PATH, "output", "edges", "edges_data.pkl")
ENRICHED_DF = os.path.join(BASE_PATH, "output", "dataframes", "enriched_df.parquet")
SUBGRAPH_FILE = os.path.join(BASE_PATH, "output", "graphs", "subgraph.graphml")
PREDICTION_OUTPUT = os.path.join(
    BASE_PATH, "output", "probes", "probes-prt-predictions.csv"
)

# Region configuration
REGION_TYPE = "city"
REGION_NAME = "Gaia"


# Filters Parser
TS_START = "2020-08-31"
TS_END = "2020-11-01"
MAX_HDOP = 7.0
MAX_SPEED = 120.0
