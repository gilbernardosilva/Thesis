import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd

from app.fcd.config.variables import FEATURES_DATA_FILE

# Validate input file
if not FEATURES_DATA_FILE or not FEATURES_DATA_FILE.endswith(".parquet"):
    raise ValueError(f"Invalid Parquet file path: {FEATURES_DATA_FILE}")

# Load Parquet with available columns from the schema
ddf = dd.read_parquet(
    FEATURES_DATA_FILE,
    engine="pyarrow",
    columns=[
        "sri",
        "road_class_encoded",
        "is_peak_hour",
        "probe_speed",
        "max_continuous_stopped_time",
        "total_stopped_time_per_edge_per_traj",
        "edge_id",
        "traj_id",
    ],
    filters=[("sri", ">=", 0)],
)

# Compute basic statistics for residential roads (road_class_encoded == 4)
residential_rows = ddf[ddf["road_class_encoded"] == 4].shape[0].compute()
logging.info(f"Total residential rows: {residential_rows}")

# Compute SRI statistics
sri_stats = ddf["sri"].describe().compute()

# Compute free-flow proportion (SRI <= 0.01) by road class
sri_free_flow = (
    ddf.groupby("road_class_encoded")["sri"]
    .apply(lambda x: (x <= 0.01).mean(), meta=("sri", "float64"))
    .compute()
)

# Compute stopped traffic proportion (SRI >= 0.99) by road class
sri_stopped = (
    ddf.groupby("road_class_encoded")["sri"]
    .apply(lambda x: (x >= 0.99).mean(), meta=("sri", "float64"))
    .compute()
)

# Debug SRI metrics
sri_mean = ddf["sri"].mean().compute()
sri_free_flow_count = (ddf["sri"] <= 0.01).sum().compute()
sri_stopped_count = (ddf["sri"] >= 0.99).sum().compute()
residential_free_flow_count = (
    ddf[(ddf["road_class_encoded"] == 4) & (ddf["sri"] <= 0.01)].shape[0].compute()
)
residential_stopped_count = (
    ddf[(ddf["road_class_encoded"] == 4) & (ddf["sri"] >= 0.99)].shape[0].compute()
)
logging.info(f"SRI mean: {sri_mean:.4f}")
logging.info(f"Free-flow SRI count (SRI <= 0.01): {sri_free_flow_count}")
logging.info(f"Stopped traffic SRI count (SRI >= 0.99): {sri_stopped_count}")
logging.info(
    f"Residential free-flow SRI count (SRI <= 0.01): {residential_free_flow_count}"
)
logging.info(
    f"Residential stopped traffic SRI count (SRI >= 0.99): {residential_stopped_count}"
)

# Sample free-flow and stopped traffic data
if sri_free_flow_count > 0:
    free_flow_sample = ddf[ddf["sri"] <= 0.01][
        [
            "sri",
            "probe_speed",
            "road_class_encoded",
            "max_continuous_stopped_time",
            "total_stopped_time_per_edge_per_traj",
        ]
    ].head(5)
    logging.info(f"Free-flow SRI sample (SRI <= 0.01):\n{free_flow_sample}")

if sri_stopped_count > 0:
    stopped_sample = ddf[ddf["sri"] >= 0.99][
        [
            "sri",
            "probe_speed",
            "road_class_encoded",
            "max_continuous_stopped_time",
            "total_stopped_time_per_edge_per_traj",
        ]
    ].head(5)
    logging.info(f"Stopped traffic SRI sample (SRI >= 0.99):\n{stopped_sample}")

# Map road class codes to names
road_class_map = {
    0: "motorway",
    1: "primary",
    2: "secondary",
    3: "tertiary",
    4: "residential",
    5: "unclassified",
    6: "service_other",
}

# Compute mean SRI by road class and peak hour
segmented_stats = ddf.groupby(["road_class_encoded"])["sri"].mean().compute()
segmented_stats.index = segmented_stats.index.map(road_class_map)

# Check low-speed data (probe_speed <= 2)
low_speed = ddf[ddf["probe_speed"] <= 2][["road_class_encoded", "is_peak_hour"]]
low_speed_counts = (
    low_speed.groupby(["road_class_encoded", "is_peak_hour"]).size().compute()
)
urban_low_speed = (
    low_speed[
        (low_speed["is_peak_hour"] == 1)
        & low_speed["road_class_encoded"].isin([1, 2, 3, 4])
    ]
    .groupby(["road_class_encoded", "is_peak_hour"])
    .size()
    .compute()
)

# Compute unique edge IDs
unique_edge_ids = ddf["edge_id"].nunique().compute()
logging.info(f"Unique edge IDs: {unique_edge_ids}")

# Compute SRI distribution by road class
bins = np.linspace(0, 1, 21)  # 20 bins: [0, 0.05), [0.05, 0.10), ..., [0.95, 1.0]
bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
distributions = {}
for code, road_class in road_class_map.items():
    try:
        sri_data = ddf[ddf["road_class_encoded"] == code]["sri"].compute()
        total_count = len(sri_data)
        logging.info(f"Total records for {road_class}: {total_count}")
        hist, _ = np.histogram(sri_data, bins=bins)
        distributions[road_class] = hist
        print(f"\nSRI Distribution for {road_class} Roads (Total: {total_count}):")
        for label, count in zip(bin_labels, hist):
            print(f"Bin {label}: {count}")
    except Exception as e:
        logging.error(f"Error calculating distribution for {road_class}: {e}")

# Compute statistics for stopped time metrics
stopped_time_stats = {}
for metric in ["max_continuous_stopped_time", "total_stopped_time_per_edge_per_traj"]:
    stopped_time_stats[metric] = ddf[metric].describe().compute()

# Validate stopped time metrics
stopped_time_validation = ddf[
    (ddf["max_continuous_stopped_time"] > 0)
    & (ddf["total_stopped_time_per_edge_per_traj"] == 0)
][
    [
        "edge_id",
        "traj_id",
        "max_continuous_stopped_time",
        "total_stopped_time_per_edge_per_traj",
    ]
].head(
    5
)
logging.info(
    f"Rows where max_continuous_stopped_time > 0 but total_stopped_time_per_edge_per_traj = 0:\n{stopped_time_validation}"
)

# Print results
print("\nSRI Statistics:")
print(sri_stats.round(3))
print("\nProportion of Free-Flow Conditions (SRI <= 0.01) by Road Class:")
print(sri_free_flow.rename(index=road_class_map).round(4))
print("\nProportion of Stopped Traffic (SRI >= 0.99) by Road Class:")
print(sri_stopped.rename(index=road_class_map).round(4))
print(f"\nUnique Edge IDs: {unique_edge_ids}")
print("\nMean SRI by Road Class:")
print(segmented_stats.round(3))
print("\nLow-Speed Data Counts (probe_speed <= 2) by Road Class and Peak Hour:")
print(
    low_speed_counts.rename_axis(index=["road_class_encoded", "is_peak_hour"])
    .reset_index()
    .assign(road_class=lambda x: x["road_class_encoded"].map(road_class_map))
    .set_index(["road_class", "is_peak_hour"])
    .drop(columns="road_class_encoded")
    .astype(int)
)
print("\nUrban Peak-Hour Low-Speed Counts (probe_speed <= 2):")
print(
    urban_low_speed.rename_axis(index=["road_class_encoded", "is_peak_hour"])
    .reset_index()
    .assign(road_class=lambda x: x["road_class_encoded"].map(road_class_map))
    .set_index(["road_class", "is_peak_hour"])
    .drop(columns="road_class_encoded")
    .astype(int)
)
print("\nStopped Time Metrics Stats:")
for metric, stats in stopped_time_stats.items():
    print(f"\n{metric} Stats:")
    print(stats.round(2))
