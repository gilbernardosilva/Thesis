import logging
import os
import time
from typing import Optional

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from distributed import Client, LocalCluster

from app.fcd.config.schemas import CLEAN_DATA_SCHEMA, INPUT_CSV_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    CLEAN_DATA_FILE,
    DASK_MEMORY_LIMIT,
    DASK_NUM_CORES,
    DASK_NUM_THREADS,
    MAX_HDOP,
    MAX_SPEED,
    PROBES_FOLDER,
    TS_END,
    TS_START,
)


def load_existing_cleaned_data(output_file: str) -> Optional[dd.DataFrame]:
    """Load previously cleaned data from a Parquet file if it exists.

    This function checks if a Parquet file containing cleaned data exists at the specified path.
    If found, it attempts to load the data into a Dask DataFrame using the provided schema.
    If the file does not exist or loading fails, it logs the appropriate message and returns None.

    Args:
        output_file (str): Path to the Parquet file containing cleaned data.

    Returns:
        Optional[dd.DataFrame]: A Dask DataFrame containing the cleaned data if the file exists and is valid,
            otherwise None.
    """
    if not os.path.exists(output_file):
        logging.info("File not found: %s. Will process raw CSVs.", output_file)
        return None

    try:
        logging.info("Loading existing cleaned data: %s", output_file)
        return dd.read_parquet(output_file, engine="pyarrow", schema=CLEAN_DATA_SCHEMA)
    except Exception as e:
        logging.warning("Failed to load cleaned data (%s): %s", output_file, e)
        return None


def process_file(file_path: str) -> Optional[dd.DataFrame]:
    """Process a single CSV file into a cleaned Dask DataFrame.

    This function reads a CSV file into a Dask DataFrame with specified block size and data type backend.
    It then passes the DataFrame to the validation and cleaning function. If any error occurs during
    reading or processing, it logs the error and returns None.

    Args:
        file_path (str): Path to the CSV file to process.

    Returns:
        Optional[dd.DataFrame]: A cleaned Dask DataFrame if processing is successful, otherwise None.
    """
    logging.info("Processing %s", os.path.basename(file_path))
    try:
        ddf = dd.read_csv(
            file_path, blocksize="128MB", dtype_backend="pyarrow", assume_missing=True
        )
        return validate_and_clean_dask(ddf, file_path)
    except Exception as e:
        logging.warning("Error processing %s: %s", file_path, e)
        return None


def validate_and_clean_dask(ddf: dd.DataFrame, file: str) -> dd.DataFrame:
    """Validate and clean a Dask DataFrame based on predefined criteria.

    This function ensures the input Dask DataFrame contains all required columns and applies a series
    of filters to clean the data. It converts the 'timestamp' column to datetime, then filters rows based
    on timestamp range, valid latitude/longitude, HDOP (Horizontal Dilution of Precision), speed, and
    non-null timestamps. It selects only the columns defined in the clean data schema and logs statistics
    about invalid rows (e.g., invalid timestamps, coordinates, or NaT values).

    Args:
        ddf (dd.DataFrame): The input Dask DataFrame to validate and clean.
        file (str): Path to the source file, used for logging.

    Returns:
        dd.DataFrame: A cleaned Dask DataFrame containing only valid rows and required columns.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    required_columns = INPUT_CSV_SCHEMA.names
    missing = [col for col in required_columns if col not in ddf.columns]
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

    df_clean = ddf[mask][CLEAN_DATA_SCHEMA.names]

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
    input_folder_path: str = PROBES_FOLDER, output_file: str = CLEAN_DATA_FILE
) -> dd.DataFrame:
    """Load and clean all CSV files in a specified folder, saving the result to a Parquet file.

    This function orchestrates the data processing pipeline. It first checks for existing cleaned data
    and returns it if available. Otherwise, it initializes a Dask LocalCluster for parallel processing,
    processes all CSV files in the input folder, concatenates the cleaned DataFrames, and saves the result
    to a Parquet file. It logs progress, errors, and timing information throughout the process. The function
    ensures the Dask client and cluster are properly shut down after execution.

    Args:
        input_folder_path (str, optional): Path to the folder containing input CSV files.
            Defaults to PROBES_FOLDER.
        output_file (str, optional): Path where the cleaned Parquet file will be saved.
            Defaults to CLEAN_DATA_FILE.

    Returns:
        dd.DataFrame: A Dask DataFrame containing the concatenated cleaned data.

    Raises:
        FileNotFoundError: If no CSV files are found in the input folder.
        RuntimeError: If no valid DataFrames are produced during cleaning.
    """
    start = time.time()

    cluster = LocalCluster(
        n_workers=DASK_NUM_CORES,
        threads_per_worker=DASK_NUM_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
    )
    client = Client(cluster)

    try:
        if (df := load_existing_cleaned_data(output_file)) is not None:
            return df

        all_files = [
            entry.path
            for entry in os.scandir(input_folder_path)
            if entry.name.endswith(".csv") and entry.is_file()
        ]
        if not all_files:
            logging.error("No CSV files found in %s", input_folder_path)
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
        ensure_directory_exists(os.path.dirname(output_file))
        with ProgressBar():
            final_ddf.to_parquet(
                output_file,
                engine="pyarrow",
                schema=CLEAN_DATA_SCHEMA,
                write_index=False,
                compression="snappy",
            )

        logging.info(
            "Cleaned data saved to %s in %.2fs", output_file, time.time() - start
        )

        return final_ddf

    finally:
        logging.info("Shutting down Dask client and cluster...")
        client.close()
        cluster.close()
