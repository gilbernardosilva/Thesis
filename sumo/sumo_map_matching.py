import logging
import os
import time

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from app.fcd.config.schemas import FILTERED_DATA_SCHEMA, MAPPED_DATA_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    DASK_MEMORY_LIMIT,
    DASK_NUM_CORES,
    DASK_NUM_THREADS,
    SUMO_FCD_PARSED_FILE,
    SUMO_MATCHED_FILE,
)
from app.fcd.map_matching import process_partition


def map_match_fcd():

    start_ts = time.time()
    logging.info("Starting map matching pipeline...")
    if os.path.exists(SUMO_MATCHED_FILE):
        logging.info(f"Output already exists at {SUMO_MATCHED_FILE}, aborting.")
        return
    cluster = LocalCluster(
        n_workers=DASK_NUM_CORES,
        threads_per_worker=DASK_NUM_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
    )
    client = Client(cluster)
    try:
        ddf = dd.read_parquet(
            SUMO_FCD_PARSED_FILE, engine="pyarrow", schema=FILTERED_DATA_SCHEMA
        )
        matched = ddf.map_partitions(process_partition)
        with ProgressBar():
            matched.to_parquet(
                SUMO_MATCHED_FILE,
                engine="pyarrow",
                write_index=False,
                compression="snappy",
                schema=MAPPED_DATA_SCHEMA,
            )
        total = ddf["traj_id"].nunique().compute()
        success = matched["traj_id"].nunique().compute()
        logging.info(
            f"Total trajectories: {total}, Successfully matched: {success}, Failed: {total - success}"
        )
        logging.info(f"Map matching completed in {time.time() - start_ts:.2f} seconds")
    except Exception as e:
        logging.error(f"Map matching pipeline failed: {e}")
    finally:
        client.close()
        cluster.close()
