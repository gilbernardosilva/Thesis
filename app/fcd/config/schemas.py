import pyarrow as pa

# Schema for input CSV files (raw probe data)
INPUT_CSV_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("timestamp", pa.float64()),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("speed", pa.float32()),
        ("hdop", pa.float32()),
        ("ground", pa.float32()),
    ]
)

# Schema for cleaned data Parquet file
CLEAN_DATA_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("speed", pa.float32()),
    ]
)

# Schema for filtered data Parquet file
FILTERED_DATA_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("speed", pa.float32()),
    ]
)

# Schema for mapped data Parquet file
MAPPED_DATA_SCHEMA = pa.schema(
    [
        ("traj_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("edge_id", pa.int64()),
        ("speed_segment", pa.float64()),
        ("length_segment", pa.float64()),
        ("speed_limit", pa.float64()),
        ("speed", pa.float64()),
    ]
)

#  Pandas dtypes for mapped data
MAPPED_DATA_PANDAS_DTYPES = {
    "traj_id": "string",
    "timestamp": "datetime64[ns]",
    "edge_id": "int64",
    "speed_segment": "float64",
    "length_segment": "float64",
    "speed_limit": "float64",
    "speed": "float64",
}


FEATURES_DATA_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ns")),
        ("edge_id", pa.int64()),
        ("traj_id", pa.string()),
        ("speed", pa.float32()),
        ("free_flow_speed", pa.float32()),
        ("speed_segment", pa.float32()),
        ("sri", pa.float32()),
        ("day_of_week", pa.int8()),
        ("hour", pa.int8()),
        ("is_peak_hour", pa.int8()),
        ("is_weekend", pa.int8()),
    ]
)
