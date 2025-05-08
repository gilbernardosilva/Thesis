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


def ensure_directory_exists(path: str):
    """Ensure the directory for the given path exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def print_n_random_rows(file_path: str, n: int = 20):
    """Print the first n rows from a Parquet file."""
    if not file_path.endswith((".parq", ".parquet", ".pq")):
        raise ValueError(f"File must be a Parquet file: {file_path}")

    logging.info(f"Reading first {n} rows from {file_path}")
    try:
        # Read with Dask for efficiency
        ddf = dd.read_parquet(file_path)

        # Print total number of rows
        total_rows = ddf.shape[0].compute()
        print(f"Total number of rows: {total_rows}")

        logging.info("Columns and dtypes:")
        print(ddf.dtypes)

        # Display first n rows using head()
        logging.info(f"First {n} rows:")
        print(ddf.head(n))

        # Read schema with pyarrow
        logging.info("Parquet schema:")
        table = pq.read_table(file_path)
        print(table.schema)

    except Exception as e:
        logging.error(f"Failed to read Parquet file {file_path}: {str(e)}")
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
