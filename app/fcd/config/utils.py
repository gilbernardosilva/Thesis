import csv
import logging
import os

import dask.dataframe as dd
import pandas as pd
import psutil
import pyarrow.parquet as pq


def count_csv_lines(file_path: str) -> int:
    """Counts the number of lines in a CSV file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f))
    except FileNotFoundError:
        print(f"Arquivo {file_path} não encontrado.")
        return 0
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return 0


def ensure_directory_exists(path: str) -> None:
    """
    Ensure the directory for the given path exists, creating it if necessary.
    """
    if not path:
        logging.error("Path is empty")
        raise ValueError("Path cannot be empty")

    dir_path = os.path.dirname(path)
    if not dir_path:
        logging.error(f"No directory component in path: {path}")
        raise ValueError(f"Path {path} has no directory component")

    try:
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            logging.error(f"Failed to create directory: {dir_path}")
            raise OSError(f"Directory {dir_path} could not be created")
        if not os.access(dir_path, os.W_OK):
            logging.error(f"No write permission for directory: {dir_path}")
            raise OSError(f"Directory {dir_path} is not writable")
        logging.debug(f"Directory {dir_path} verified or created")
    except OSError as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        raise


def print_n_first_rows(file_path: str, n: int = 20):
    """Print the first n rows from a Parquet file."""
    if not file_path.endswith((".parq", ".parquet", ".pq")):
        raise ValueError(f"File must be a Parquet file: {file_path}")

    logging.info(f"Reading first {n} rows from {file_path}")
    try:
        ddf = dd.read_parquet(file_path)

        total_rows = ddf.shape[0].compute()
        logging.info(f"Total number of rows: {total_rows}")

        logging.info("Columns and dtypes:")
        logging.info(ddf.dtypes)

        logging.info(f"First {n} rows:")
        logging.info(ddf.head(n))

    except Exception as e:
        logging.error(f"Failed to read Parquet file {file_path}: {str(e)}")
        raise


def export_n_first_rows_to_csv(file_path: str, output_csv_path: str, n: int = 100):
    """Export the first n rows from a Parquet file to a CSV file."""
    if not file_path.endswith((".parq", ".parquet", ".pq")):
        raise ValueError(f"File must be a Parquet file: {file_path}")

    logging.info(f"Reading first {n} rows from {file_path}")
    try:
        ddf = dd.read_parquet(file_path)

        total_rows = ddf.shape[0].compute()
        logging.info(f"Total number of rows: {total_rows}")

        logging.info("Columns and dtypes:")
        logging.info(ddf.dtypes)

        logging.info(f"Exporting first {n} rows to {output_csv_path}")
        df_head = ddf.head(n, compute=True)

        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        df_head.to_csv(
            output_csv_path,
            index=False,
            float_format="%.6f",
            encoding="utf-8",
        )
        logging.info(f"Successfully exported {n} rows to {output_csv_path}")

    except Exception as e:
        logging.error(
            f"Failed to process Parquet file {file_path} or write to {output_csv_path}: {str(e)}"
        )
        raise


def print_summary(mapped_df: pd.DataFrame, enriched_df: pd.DataFrame):
    """Print a summary of the processed data."""
    logging.info("\n=== SUMMARY ===")
    probe_ids = (
        mapped_df["probe_id"].nunique() if "probe_id" in mapped_df.columns else "N/A"
    )
    logging.info(f"Unique probe IDs: {probe_ids}")
    logging.info(f"Unique segments (raw): {mapped_df['segment'].nunique()}")
    logging.info(f"Unique segments (aggregated): {enriched_df['segment'].nunique()}")
    logging.info(f"Total time intervals: {enriched_df['timestamp'].nunique()}")
    logging.info(f"Total final records: {len(enriched_df)}")

    logging.info("\n=== Enriched Data Preview ===")
    logging.info("\n" + enriched_df.head(10).to_string())
