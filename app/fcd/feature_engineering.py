import logging
import os
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from app.fcd.config.schemas import FEATURES_DATA_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    FEATURES_DATA_FILE,
    MAPPED_DATA_FILE,
    OFF_PEAK_HOURS,
)


def calculate_sri(
    df,
    speed_column="speed",
    free_flow_column="free_flow_speed",
):
    """Calculate Speed Reduction Index (SRI) using free_flow_speed."""
    df[free_flow_column] = df[free_flow_column].clip(lower=1e-6)
    df[speed_column] = df[speed_column].clip(lower=0)

    sri = 1 - df[speed_column] / df[free_flow_column]
    return sri.clip(lower=0, upper=1).replace([np.inf, -np.inf], np.nan)


def compute_free_flow_speed(ddf):
    """Compute 95th percentile free flow speed during off-peak hours using Pandas."""
    night_ddf = ddf[ddf["timestamp"].dt.hour.isin(OFF_PEAK_HOURS)][["edge_id", "speed"]]
    logging.info(f"Rows in night_ddf: {night_ddf.shape[0].compute()}")

    night_pdf = night_ddf.compute()

    free_flow_pdf = (
        night_pdf.groupby("edge_id")["speed"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"speed": "free_flow_speed"})
    )

    free_flow_pdf = free_flow_pdf.astype(
        {"edge_id": "int64", "free_flow_speed": "float32"}
    )

    free_flow_ddf = dd.from_pandas(free_flow_pdf, npartitions=2).persist()

    unique_edge_ids = free_flow_ddf["edge_id"].nunique().compute()
    logging.info(f"Number of unique edge_ids in free_flow_ddf: {unique_edge_ids}")
    logging.info(f"Rows in free_flow_ddf: {free_flow_ddf.shape[0].compute()}")
    logging.info(
        f"Free-flow speed range: {free_flow_ddf['free_flow_speed'].min().compute()} to "
        f"{free_flow_ddf['free_flow_speed'].max().compute()}"
    )
    logging.info(
        f"Sample edge_id in free_flow_ddf: {free_flow_ddf['edge_id'].head().tolist()}"
    )

    return free_flow_ddf


def aggregate_and_calculate(ddf, free_flow_ddf):
    """Aggregate data and calculate features."""
    required_columns = [
        "traj_id",
        "edge_id",
        "timestamp",
        "speed",
        "speed_segment",
    ]

    missing_columns = [col for col in required_columns if col not in ddf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    probe_ddf = ddf[required_columns].astype({"edge_id": "int64"})
    logging.info(f"Rows in probe_ddf: {probe_ddf.shape[0].compute()}")
    logging.info(
        f"Unique edge_ids in probe_ddf: {probe_ddf['edge_id'].nunique().compute()}"
    )
    logging.info(f"Sample edge_id in probe_ddf: {probe_ddf['edge_id'].head().tolist()}")

    free_flow_ddf = free_flow_ddf.drop_duplicates(subset=["edge_id"])
    logging.info(
        f"Rows in free_flow_ddf after deduplication: {free_flow_ddf.shape[0].compute()}"
    )

    merged_ddf = probe_ddf.merge(
        free_flow_ddf, on="edge_id", how="left", suffixes=("", "_free_flow")
    ).persist()
    logging.info(f"Rows in merged_ddf: {merged_ddf.shape[0].compute()}")

    if merged_ddf["speed_segment"].isna().any().compute():
        raise ValueError("Valores nulos encontrados em speed_segment")
    merged_ddf["speed_segment"] = merged_ddf["speed_segment"].clip(lower=5, upper=200)

    # Substituir NaN em free_flow_speed por speed_segment
    merged_ddf["free_flow_speed"] = merged_ddf["free_flow_speed"].fillna(
        merged_ddf["speed_segment"]
    )

    merged_ddf["sri"] = calculate_sri(merged_ddf)
    sri_extreme_count = merged_ddf["sri"].isin([0, 1]).sum().compute()
    logging.info(f"Registros com SRI extremo (0 ou 1): {sri_extreme_count}")

    merged_ddf["day_of_week"] = merged_ddf["timestamp"].dt.dayofweek.astype("int8")
    merged_ddf["hour"] = merged_ddf["timestamp"].dt.hour.astype("int8")
    merged_ddf["is_peak_hour"] = (
        merged_ddf["timestamp"].dt.hour.isin([7, 8, 17, 18]).astype("int8")
    )
    merged_ddf["is_weekend"] = (
        merged_ddf["timestamp"].dt.dayofweek.isin([5, 6]).astype("int8")
    )

    nan_sri_count = merged_ddf["sri"].isna().sum().compute()
    logging.info(f"Number of NaN SRI values: {nan_sri_count}")

    logging.info(f"Data types in merged_ddf before writing: {merged_ddf.dtypes}")

    return merged_ddf


def feature_engineering():
    """Execute feature engineering pipeline."""
    start_ts = time.time()
    logging.info("Starting feature engineering pipeline...")

    if os.path.exists(FEATURES_DATA_FILE):
        logging.info(
            "Output file already exists, skipping feature engineering pipeline."
        )
        return

    ensure_directory_exists(os.path.dirname(FEATURES_DATA_FILE))

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="3GB")
    client = Client(cluster)
    try:
        ddf = dd.read_parquet(MAPPED_DATA_FILE, engine="pyarrow")
        logging.info(f"Input rows: {ddf.shape[0].compute()}")
        logging.info(f"Sample edge_id in input ddf: {ddf['edge_id'].head().tolist()}")

        free_flow_ddf = compute_free_flow_speed(ddf)
        result_ddf = aggregate_and_calculate(ddf, free_flow_ddf)

        logging.info(f"Output rows before writing: {result_ddf.shape[0].compute()}")
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
