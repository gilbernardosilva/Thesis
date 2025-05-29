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
        ("traj_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("speed", pa.float32()),
    ]
)


# Schema for filtered data (assumed)
FILTERED_DATA_SCHEMA = pa.schema(
    [
        ("traj_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("speed", pa.float64()),
    ]
)

# Schema for mapped data
MAPPED_DATA_SCHEMA = pa.schema(
    [
        ("traj_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("edge_id", pa.int64()),
        ("speed_segment", pa.float64()),
        ("length_segment", pa.float64()),
        ("probe_speed", pa.float64()),
        ("num_lanes", pa.int8()),
        ("road_class", pa.string()),
    ]
)

# Schema for features data
FEATURES_DATA_SCHEMA = pa.schema(
    [
        ("edge_id", pa.int64()),
        ("traj_id", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("probe_speed", pa.float32()),
        ("speed_segment", pa.float32()),
        ("length_segment", pa.float32()),
        ("road_class", pa.string()),
        ("road_class_encoded", pa.int8()),
        ("sri", pa.float32()),
        ("day_of_week", pa.int8()),
        ("hour", pa.int8()),
        ("is_peak_hour", pa.int8()),
        ("is_weekend", pa.int8()),
        ("avg_speed_segment", pa.float64()),
        ("speed_bin_0", pa.float64()),
        ("speed_bin_1", pa.float64()),
        ("speed_bin_2", pa.float64()),
        ("speed_bin_3", pa.float64()),
        ("speed_bin_4", pa.float64()),
        (
            "max_continuous_stopped_time",
            pa.float32(),
        ),
        (
            "total_stopped_time_per_edge_per_traj",
            pa.float32(),
        ),
    ]
)

# Schema for mapped road classes
ROAD_CLASS_ARRAY = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "service_other",
]

SUMO_ACCIDENTS_SCHEMA = pa.schema(
    [
        ("vehicle_id", pa.string()),
        ("start_time", pa.timestamp("ns")),
        ("end_time", pa.timestamp("ns")),
        ("duration", pa.int32()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("edge_id", pa.int64()),
        ("sumo_edge_id", pa.string()),
    ]
)
