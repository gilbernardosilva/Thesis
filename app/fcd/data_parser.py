import logging
import os
import time
from typing import Optional

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from distributed import Client, LocalCluster

from app.fcd.config.schemas import (
    CLEAN_INPUT_SCHEMA,
    CLEAN_OUTPUT_SCHEMA,
    CLEAN_REQUIRED_COLUMNS,
)
from app.fcd.config.variables import (
    CLEAN_DF,
    MAX_HDOP,
    MAX_SPEED,
    PROBES_FOLDER,
    TS_END,
    TS_START,
)


def load_existing_cleaned_data(output_file: str) -> Optional[dd.DataFrame]:
    """Load existing cleaned data if present."""
    if not os.path.exists(output_file):
        logging.info("File not found: %s. Will process raw CSVs.", output_file)
        return None

    try:
        logging.info("Loading existing cleaned data: %s", output_file)
        return dd.read_parquet(output_file)
    except Exception as e:
        logging.warning("Failed to load cleaned data (%s): %s", output_file, e)
        return None


def process_file(file_path: str) -> Optional[dd.DataFrame]:
    """Process a single CSV file."""
    logging.info("Processing %s", os.path.basename(file_path))
    try:
        ddf = dd.read_csv(
            file_path, blocksize="128MB", dtype=CLEAN_INPUT_SCHEMA, assume_missing=True
        )
        return validate_and_clean_dask(ddf, file_path)
    except Exception as e:
        logging.warning("Error processing %s: %s", file_path, e)
        return None


def validate_and_clean_dask(ddf: dd.DataFrame, file: str) -> dd.DataFrame:
    """Validate and clean a Dask DataFrame."""
    missing = [col for col in CLEAN_REQUIRED_COLUMNS if col not in ddf.columns]
    if missing:
        raise ValueError(f"Missing columns in {file}: {missing}")

    ddf["timestamp"] = dd.to_datetime(ddf["timestamp"], unit="s", errors="coerce")

    mask = (
        (ddf["timestamp"] > TS_START)
        & (ddf["timestamp"] < TS_END)
        & ddf["latitude"].between(-90, 90)
        & ddf["longitude"].between(-180, 180)
        & (ddf["hdop"] <= MAX_HDOP)
        & ddf["speed"].between(0, MAX_SPEED)
        & ddf["timestamp"].notnull()
    )

    df_clean = ddf[mask].drop(columns=["hdop", "ground"])
    df_clean = df_clean.astype(CLEAN_OUTPUT_SCHEMA)

    invalid_ts, invalid_coords, nat_count, total_rows, clean_rows = dask.compute(
        (~((ddf["timestamp"] > TS_START) & (ddf["timestamp"] < TS_END))).sum(),
        (
            ~(ddf["latitude"].between(-90, 90) & ddf["longitude"].between(-180, 180))
        ).sum(),
        df_clean["timestamp"].isna().sum(),
        ddf.shape[0],
        df_clean.shape[0],
    )

    logging.info("Rows before cleaning (%s): %d", file, total_rows)
    logging.info("Rows after cleaning (%s): %d", file, clean_rows)
    logging.info(
        "Removed: %d (%.2f%%)",
        total_rows - clean_rows,
        ((total_rows - clean_rows) / total_rows) * 100,
    )

    if invalid_ts:
        logging.warning("Invalid timestamps in %s: %d rows", file, invalid_ts)
    if invalid_coords:
        logging.warning("Invalid coordinates in %s: %d rows", file, invalid_coords)
    if nat_count:
        logging.warning("NaT timestamps in %s: %d rows", file, nat_count)

    return df_clean


def load_and_clean_data(
    folder_path: str = PROBES_FOLDER, output_file: str = CLEAN_DF
) -> dd.DataFrame:
    """Load and clean all CSV files in a folder."""
    start = time.time()

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="auto",
    )

    with Client(cluster):
        if (df := load_existing_cleaned_data(output_file)) is not None:
            return df

        all_files = [
            entry.path
            for entry in os.scandir(folder_path)
            if entry.name.endswith(".csv") and entry.is_file()
        ]
        if not all_files:
            raise FileNotFoundError("No CSV files found in the folder.")

        logging.info("%d files found.", len(all_files))

        cleaned_ddfs = []
        total_files = len(all_files)

        for idx, file in enumerate(all_files, start=1):
            logging.info(
                "Processing file %d/%d: %s", idx, total_files, os.path.basename(file)
            )
            df = process_file(file)
            if df is not None:
                cleaned_ddfs.append(df)

        if not cleaned_ddfs:
            raise RuntimeError("No valid DataFrames produced during cleaning.")

        logging.info("Concatenating cleaned DataFrames...")
        final_ddf = dd.concat(cleaned_ddfs, ignore_index=True)

        logging.info("Saving cleaned data to Parquet...")
        with ProgressBar():
            final_ddf.to_parquet(output_file, engine="pyarrow", write_index=False)

        logging.info(
            "Cleaned data saved to %s in %.2fs", output_file, time.time() - start
        )

        return final_ddf
