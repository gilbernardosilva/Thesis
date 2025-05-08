import logging
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd

from app.fcd.config.variables import ENRICHED_DF, MAPPED_DF


def feature_engineer():
    start = time.time()
    logging.info("Loading mapped FCD data from Parquet…")
    df = dd.read_parquet(MAPPED_DF)
    logging.info(f"Initial number of partitions: {df.npartitions}")
    sample = df.head(10)
    logging.info(
        f"After loading data: Duplicated indices in sample: {sample.index.duplicated().any()}"
    )
    logging.info(f"After loading data: Sample index head: {sample.index.tolist()}")
    logging.info("Resetting index after loading data…")
    df = df.reset_index(drop=True)
    sample = df.head(10)
    logging.info(
        f"After initial reset: Duplicated indices in sample: {sample.index.duplicated().any()}"
    )
    logging.info(f"After initial reset: Sample index head: {sample.index.tolist()}")
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    df["time_window"] = df["timestamp"].dt.floor("5min")
    logging.info("Calculating aggregated features per segment and time window…")
    grouped = df.groupby(["segment", "time_window"])
    avg_speed = grouped["speed"].mean().rename("avg_speed")
    std_speed = grouped["speed"].std().rename("std_speed")
    median_speed = grouped["speed"].median().rename("median_speed")
    vehicle_count = grouped.size().rename("vehicle_count")
    features = dd.concat([avg_speed, std_speed, median_speed, vehicle_count], axis=1)
    features["hour"] = features.index.get_level_values("time_window").hour
    features["day_of_week"] = features.index.get_level_values("time_window").dayofweek
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
    features["time_of_day_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["time_of_day_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    logging.info("Persisting after time features…")
    features = features.persist()
    sample = features.head(10)
    logging.info(
        f"After time features: Duplicated indices in sample: {sample.index.duplicated().any()}"
    )
    logging.info(f"After time features: Sample index head: {sample.index.tolist()}")
    logging.info("Calculating lag features…")
    features = features.sort_index(level=["segment", "time_window"])
    features["speed_lag1"] = grouped["speed"].mean().shift(1)
    features["speed_lag2"] = grouped["speed"].mean().shift(2)
    features["speed_diff"] = grouped["speed"].mean().diff()
    logging.info("Persisting after lag features…")
    features = features.persist()
    sample = features.head(10)
    logging.info(
        f"After lag features: Duplicated indices in sample: {sample.index.duplicated().any()}"
    )
    logging.info(f"After lag features: Sample index head: {sample.index.tolist()}")
    logging.info("Defining congestion label based on historical speed percentile…")
    historical_speeds = dd.read_parquet("./data/historical_speeds.parquet")
    features = features.merge(
        historical_speeds[["speed_p25"]], left_on="segment", right_index=True
    )
    features["speed_ratio"] = features["avg_speed"] / features["speed_p25"]
    features["congestion"] = (features["avg_speed"] < features["speed_p25"]).astype(
        "int8"
    )
    congestion_dist = features["congestion"].value_counts(normalize=True).compute()
    logging.info(f"Congestion distribution: {congestion_dist.to_dict()}")
    speed_stats = features["avg_speed"].describe().compute()
    logging.info(f"Average speed statistics: {speed_stats.to_dict()}")
    logging.info("Repartitioning DataFrame with partition size 500MB…")
    features = features.repartition(partition_size="500MB")
    logging.info(f"Number of partitions after repartition: {features.npartitions}")
    sample = features.head(10)
    logging.info(
        f"After repartition: Duplicated indices in sample: {sample.index.duplicated().any()}"
    )
    logging.info(f"After repartition: Sample index head: {sample.index.tolist()}")
    logging.info("Saving enriched FCD dataframe to Parquet…")
    enriched_df = features.reset_index()[
        [
            "segment",
            "time_window",
            "avg_speed",
            "std_speed",
            "median_speed",
            "vehicle_count",
            "hour",
            "day_of_week",
            "is_weekend",
            "time_of_day_sin",
            "time_of_day_cos",
            "speed_lag1",
            "speed_lag2",
            "speed_diff",
            "speed_ratio",
            "congestion",
        ]
    ]
    enriched_df.to_parquet(ENRICHED_DF, engine="pyarrow", compression="snappy")
    logging.info("Enrichment completed.")
    logging.info(f"Finished enrichment in {time.time() - start:.2f} seconds")
