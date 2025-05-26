import logging
import os
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from app.fcd.config.schemas import FEATURES_DATA_SCHEMA, ROAD_CLASS_ARRAY
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    DASK_MEMORY_LIMIT,
    DASK_NUM_CORES,
    DASK_NUM_THREADS,
    FEATURES_DATA_FILE,
    MAPPED_DATA_FILE,
    OFF_PEAK_HOURS,
    PEAK_HOURS,
)


def calculate_sri(df, speed_column="probe_speed", free_flow_column="free_flow_speed"):
    """Calculate the Speed Reduction Index (SRI) for a DataFrame.

    This function computes the SRI, which measures the reduction in probe speed relative to the
    free-flow speed, clipped between 0 and 1. It determines an effective free-flow speed by using
    the provided free-flow speed, falling back to the speed segment if the free-flow speed is too low
    compared to predefined minimums (80 km/h for motorways, 30 km/h for others). It logs the number
    of rows using the fallback and the mean SRI.

    Args:
        df: The input DataFrame containing speed and road class data.
        speed_column (str, optional): Name of the column with probe speeds. Defaults to "probe_speed".
        free_flow_column (str, optional): Name of the column with free-flow speeds. Defaults to "free_flow_speed".

    Returns:
        The SRI values as a Series, clipped between 0 and 1, with invalid values (inf, -inf, NaN) replaced by NaN.
    """
    min_free_flow = df["road_class_encoded"].map(
        lambda x: 80.0 if x == 0 else 30.0, meta=("min_free_flow", "float32")
    )
    effective_free_flow = (
        df[free_flow_column]
        .where(
            (df[free_flow_column] >= df["speed_segment"])
            | (df["speed_segment"] < min_free_flow),
            df["speed_segment"],
        )
        .clip(lower=1e-6)
    )

    fallback_count = (
        (
            (df[free_flow_column] < df["speed_segment"])
            & (df["speed_segment"] >= min_free_flow)
        )
        .sum()
        .compute()
    )
    logging.info(f"Rows using speed_segment as free_flow_speed: {fallback_count}")

    probe_speed = df[speed_column].clip(lower=0)
    sri = (1 - probe_speed / effective_free_flow).clip(lower=0, upper=1)
    sri = sri.mask((sri == np.inf) | (sri == -np.inf) | sri.isna(), np.nan)

    sri_mean = sri.mean().compute()
    logging.info(f"SRI mean: {sri_mean}")

    return sri


def compute_free_flow_speed(ddf):
    """Compute free-flow speeds for each edge based on off-peak hour data.

    This function filters the input Dask DataFrame to include only off-peak hour data, computes
    the 90th percentile speed for motorways and 80th percentile for other road classes as the
    free-flow speed, and adjusts these speeds based on road class constraints (e.g., minimum 80 km/h
    for motorways, maximum 40 km/h for residential roads). It logs statistics about the free-flow
    speeds and the number of edges where the speed segment is used as a fallback.

    Args:
        ddf: The input Dask DataFrame containing probe speeds, edge IDs, and road classes.

    Returns:
        A persisted Dask DataFrame with columns 'edge_id' and 'free_flow_speed'.
    """
    night_ddf = ddf[ddf["timestamp"].dt.hour.isin(OFF_PEAK_HOURS)][
        ["edge_id", "probe_speed", "road_class", "speed_segment"]
    ]
    counts = night_ddf.groupby("edge_id").size().compute()
    valid_edge_ids = counts[counts >= 1].index
    logging.info(f"Edge IDs after counts >= 1: {len(valid_edge_ids)}")

    night_ddf = night_ddf[night_ddf["edge_id"].isin(valid_edge_ids)]

    night_pdf = night_ddf.compute()
    road_class_mapping = {cls: idx for idx, cls in enumerate(ROAD_CLASS_ARRAY)}
    night_pdf["road_class_encoded"] = night_pdf["road_class"].map(road_class_mapping)

    free_flow_pdf = (
        night_pdf.groupby("edge_id")
        .apply(
            lambda x: x["probe_speed"].quantile(
                0.90 if x["road_class_encoded"].iloc[0] == 0 else 0.8
            )
        )
        .reset_index(name="free_flow_speed")
    )
    free_flow_pdf = free_flow_pdf.astype(
        {"edge_id": "int64", "free_flow_speed": "float32"}
    )

    free_flow_pdf = free_flow_pdf.merge(
        night_pdf[["edge_id", "road_class_encoded", "speed_segment"]].drop_duplicates(
            subset=["edge_id"]
        ),
        on="edge_id",
        how="left",
    )

    def adjust_free_flow_speed(row):
        free_flow = row["free_flow_speed"]
        if row["road_class_encoded"] == 0:
            return max(free_flow, 80.0)
        elif row["road_class_encoded"] == 4:
            return min(free_flow, 40.0)
        return free_flow

    free_flow_pdf["free_flow_speed"] = free_flow_pdf.apply(
        adjust_free_flow_speed, axis=1
    )

    replaced_edges = (
        free_flow_pdf["free_flow_speed"] < free_flow_pdf["speed_segment"]
    ).sum()
    logging.info(
        f"Number of edge_ids where free_flow_speed is replaced by speed_segment: {replaced_edges}"
    )

    free_flow_pdf["free_flow_speed"] = free_flow_pdf.apply(
        lambda row: (
            row["speed_segment"]
            if row["free_flow_speed"] < row["speed_segment"]
            else row["free_flow_speed"]
        ),
        axis=1,
    )

    logging.info(
        f"Motorway free_flow_speed min: {free_flow_pdf[free_flow_pdf['road_class_encoded'] == 0]['free_flow_speed'].min()}"
    )
    logging.info(
        f"Residential free_flow_speed stats:\n{free_flow_pdf[free_flow_pdf['road_class_encoded'] == 4]['free_flow_speed'].describe()}"
    )

    free_flow_ddf = dd.from_pandas(
        free_flow_pdf[["edge_id", "free_flow_speed"]], npartitions=2
    ).persist()
    logging.info(f"Columns in free_flow_ddf: {list(free_flow_ddf.columns)}")
    logging.info(
        f"Unique edge_ids in free_flow_ddf: {free_flow_ddf['edge_id'].nunique().compute()}"
    )
    return free_flow_ddf


def compute_aggregate_features(ddf):
    """Compute aggregated features for each edge and trajectory.

    This function processes the input Dask DataFrame to compute features such as average speed per
    edge, speed distribution across predefined bins, maximum continuous stopped time, and total
    stopped time per edge and trajectory. It identifies stopped vehicles (speed ≤ 1 km/h) and
    calculates stop durations, handling missing values appropriately.

    Args:
        ddf: The input Dask DataFrame containing probe speeds, edge IDs, trajectory IDs, and timestamps.

    Returns:
        A persisted Dask DataFrame with aggregated features for each edge and trajectory.
    """
    bins = [0, 20, 40, 60, 80, 100]
    labels = [f"speed_bin_{i}" for i in range(len(bins) - 1)]

    base_pdf = ddf[["edge_id", "traj_id"]].drop_duplicates().compute()

    pdf = ddf[["edge_id", "probe_speed", "traj_id", "timestamp"]].compute()
    pdf = pdf.sort_values(["traj_id", "timestamp"])

    pdf["time_diff"] = (
        pdf.groupby("traj_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    )

    pdf["is_stopped"] = (pdf["probe_speed"] <= 1).astype(int)

    pdf["stop_group"] = (
        pdf.groupby("traj_id")["is_stopped"].cumsum().where(pdf["is_stopped"] == 1, 0)
    )
    pdf["stop_group"] = pdf.groupby("traj_id")["stop_group"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    stop_durations = (
        pdf[pdf["is_stopped"] == 1]
        .groupby(["edge_id", "traj_id", "stop_group"])["time_diff"]
        .sum()
        .reset_index(name="stop_duration")
    )

    max_stopped_per_traj = (
        stop_durations.groupby(["edge_id", "traj_id"])["stop_duration"]
        .max()
        .reset_index(name="max_continuous_stopped_time")
    )

    total_stopped_per_traj = (
        stop_durations.groupby(["edge_id", "traj_id"])["stop_duration"]
        .sum()
        .reset_index(name="total_stopped_time_per_edge_per_traj")
    )

    agg_pdf = pdf.groupby("edge_id")["probe_speed"].mean().reset_index()
    agg_pdf.columns = ["edge_id", "avg_speed_segment"]

    pdf["speed_bin"] = pd.cut(
        pdf["probe_speed"], bins=bins, labels=labels, include_lowest=True
    )
    speed_dist_pdf = (
        pdf.groupby(["edge_id", "speed_bin"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    speed_dist_pdf = speed_dist_pdf.div(
        speed_dist_pdf.sum(axis=1), axis=0
    ).reset_index()
    speed_dist_pdf.columns = ["edge_id"] + labels

    agg_pdf = (
        base_pdf.merge(agg_pdf, on="edge_id", how="left")
        .merge(max_stopped_per_traj, on=["edge_id", "traj_id"], how="left")
        .merge(speed_dist_pdf, on="edge_id", how="left")
        .merge(total_stopped_per_traj, on=["edge_id", "traj_id"], how="left")
    )

    agg_pdf["max_continuous_stopped_time"] = agg_pdf[
        "max_continuous_stopped_time"
    ].fillna(0)
    agg_pdf["total_stopped_time_per_edge_per_traj"] = agg_pdf[
        "total_stopped_time_per_edge_per_traj"
    ].fillna(0)
    for label in labels:
        agg_pdf[label] = agg_pdf[label].fillna(0)

    agg_ddf = dd.from_pandas(agg_pdf, npartitions=2).persist()
    return agg_ddf


def aggregate_and_calculate(ddf, free_flow_ddf, agg_ddf):
    """Aggregate data and calculate final features for the dataset.

    This function merges probe data with free-flow speeds and aggregated features, validates required
    columns, and computes additional features such as SRI, day of week, hour, peak hour status, and
    weekend status. It filters out invalid trajectory IDs and rows with excessive stopped times,
    encodes road classes, and logs statistics about SRI values, urban peak-hour rows, and low-speed
    counts.

    Args:
        ddf: The input Dask DataFrame with probe data.
        free_flow_ddf: Dask DataFrame with free-flow speeds per edge.
        agg_ddf: Dask DataFrame with aggregated features per edge and trajectory.

    Returns:
        A Dask DataFrame with the final feature set, conforming to FEATURES_DATA_SCHEMA.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    required_columns = [
        "traj_id",
        "edge_id",
        "timestamp",
        "probe_speed",
        "speed_segment",
        "length_segment",
        "road_class",
    ]
    missing_columns = [col for col in required_columns if col not in ddf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    probe_ddf = ddf[required_columns].astype({"edge_id": "int64"})
    logging.info(f"Rows in probe_ddf: {probe_ddf.shape[0].compute()}")
    logging.info(
        f"Unique edge_ids in probe_ddf: {probe_ddf['edge_id'].nunique().compute()}"
    )

    merged_ddf = probe_ddf.merge(free_flow_ddf, on="edge_id", how="left").merge(
        agg_ddf, on=["edge_id", "traj_id"], how="left"
    )
    merged_ddf = merged_ddf.reset_index(drop=True).repartition(npartitions=4).persist()
    logging.info(f"Rows in merged_ddf: {merged_ddf.shape[0].compute()}")
    logging.info(
        f"Unique edge_ids in merged_ddf: {merged_ddf['edge_id'].nunique().compute()}"
    )

    merged_ddf = merged_ddf[merged_ddf["traj_id"] != "missing_traj_id"].persist()
    logging.info(
        f"Rows after removing invalid traj_id: {merged_ddf.shape[0].compute()}"
    )

    road_class_mapping = {cls: idx for idx, cls in enumerate(ROAD_CLASS_ARRAY)}
    logging.info(f"Road class mapping: {road_class_mapping}")
    unique_road_classes = set(merged_ddf["road_class"].unique().compute())
    missing_classes = unique_road_classes - set(ROAD_CLASS_ARRAY)
    if missing_classes:
        logging.warning(f"Road classes not in ROAD_CLASS_ARRAY: {missing_classes}")
    merged_ddf["road_class_encoded"] = (
        merged_ddf["road_class"]
        .map(road_class_mapping, meta=("road_class_encoded", "int8"))
        .fillna(5)
        .astype("int8")
    )

    merged_ddf["sri"] = calculate_sri(merged_ddf).clip(upper=0.999)
    sri_extreme_count = merged_ddf["sri"].isin([0, 0.999]).sum().compute()
    logging.info(f"Records with extreme SRI (0 or 0.999): {sri_extreme_count}")

    sri_extreme_by_class = (
        merged_ddf[merged_ddf["sri"] >= 0.999]
        .groupby("road_class_encoded")
        .size()
        .compute()
    )
    logging.info(f"SRI=1.0 counts by road class: {sri_extreme_by_class}")

    merged_ddf["day_of_week"] = merged_ddf["timestamp"].dt.dayofweek.astype("int8")
    merged_ddf["hour"] = merged_ddf["timestamp"].dt.hour.astype("int8")
    merged_ddf["is_peak_hour"] = (
        merged_ddf["timestamp"].dt.hour.isin(PEAK_HOURS).astype("int8")
    )
    merged_ddf["is_weekend"] = (
        merged_ddf["timestamp"].dt.dayofweek.isin([5, 6]).astype("int8")
    )

    urban_peak_rows = (
        merged_ddf[
            (merged_ddf["is_peak_hour"] == 1)
            & merged_ddf["road_class_encoded"].isin([1, 2, 3, 4])
        ]
        .shape[0]
        .compute()
    )
    logging.info(f"Urban peak-hour rows before filtering: {urban_peak_rows}")

    idle_stop_mask = merged_ddf["max_continuous_stopped_time"] > 60
    merged_ddf = merged_ddf[~idle_stop_mask].persist()
    idle_stop_count = idle_stop_mask.sum().compute()
    logging.info(f"Filtered idle stop rows: {idle_stop_count}")

    urban_peak_rows_after = (
        merged_ddf[
            (merged_ddf["is_peak_hour"] == 1)
            & merged_ddf["road_class_encoded"].isin([1, 2, 3, 4])
        ]
        .shape[0]
        .compute()
    )
    logging.info(f"Urban peak-hour rows after filtering: {urban_peak_rows_after}")

    low_speed_counts = (
        merged_ddf[merged_ddf["probe_speed"] <= 2]
        .groupby(["road_class_encoded", "is_peak_hour"])
        .size()
        .compute()
    )
    logging.info(
        f"Low-speed probe counts by road class and peak hour: {low_speed_counts}"
    )

    final_edge_ids = merged_ddf["edge_id"].nunique().compute()
    final_rows = merged_ddf.shape[0].compute()
    logging.info(f"Final unique edge_ids: {final_edge_ids}")
    logging.info(f"Final rows: {final_rows}")

    output_columns = [field.name for field in FEATURES_DATA_SCHEMA]
    result_ddf = merged_ddf[output_columns]
    return result_ddf


def feature_engineering():
    """Run the feature engineering pipeline to process and save features.

    This function orchestrates the feature engineering process by loading mapped data, computing
    free-flow speeds and aggregated features, merging them with probe data, and calculating final
    features. It uses a Dask LocalCluster for parallel processing, saves the results to a Parquet
    file, and logs progress and timing. If the output file already exists, the pipeline is skipped.

    Returns:
        None

    Raises:
        Exception: If any step in the pipeline (loading, processing, or saving) fails.
    """
    start_ts = time.time()
    logging.info("Starting feature engineering pipeline...")
    if os.path.exists(FEATURES_DATA_FILE):
        logging.info("Features data file already exists. Skipping pipeline.")
        return

    ensure_directory_exists(os.path.dirname(FEATURES_DATA_FILE))

    cluster = LocalCluster(
        n_workers=DASK_NUM_CORES,
        threads_per_worker=DASK_NUM_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
    )
    client = Client(cluster, timeout="300s")

    try:
        ddf = dd.read_parquet(MAPPED_DATA_FILE, engine="pyarrow")
        logging.info(f"Input rows: {ddf.shape[0].compute()}")
        logging.info(f"Input unique edge_ids: {ddf['edge_id'].nunique().compute()}")
        free_flow_ddf = compute_free_flow_speed(ddf).persist()
        agg_ddf = compute_aggregate_features(ddf).persist()
        result_ddf = aggregate_and_calculate(ddf, free_flow_ddf, agg_ddf)
        logging.info(f"Output rows: {result_ddf.shape[0].compute()}")
        with ProgressBar():
            result_ddf.to_parquet(
                FEATURES_DATA_FILE,
                engine="pyarrow",
                compression="snappy",
                write_index=False,
                schema=FEATURES_DATA_SCHEMA,
            )
        logging.info(
            f"Feature engineering completed in {time.time() - start_ts:.2f} seconds."
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    feature_engineering()
