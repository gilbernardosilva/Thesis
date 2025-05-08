CLEAN_REQUIRED_COLUMNS = [
    "id",
    "timestamp",
    "latitude",
    "longitude",
    "speed",
]

CLEAN_INPUT_SCHEMA = {
    "id": "string",
    "timestamp": "float64",
    "latitude": "float32",
    "longitude": "float32",
    "speed": "float32",
    "hdop": "float32",
    "ground": "float32",
}

CLEAN_OUTPUT_SCHEMA = {
    "id": "string",
    "timestamp": "datetime64[ns]",
    "latitude": "float32",
    "longitude": "float32",
    "speed": "float32",
}

MAPPED_OUTPUT_SCHEMA = {
    "traj_id": "string",
    "timestamp": "datetime64[ns]",
    "edge_id": "int64",
    "speed_segment": "float64",
    "length_segment": "float64",
    "speed_limit": "float64",
    "base_speed": "float64",
}
