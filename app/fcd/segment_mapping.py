import asyncio
import logging
import os
import time
from typing import List, Optional

import aiohttp
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from distributed import Client, LocalCluster

from app.fcd.config.schemas import CLEAN_REQUIRED_COLUMNS, MAPPED_OUTPUT_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import FILTERED_DF, MAPPED_DF

CONCURRENCY_LIMIT = 10


def extract_segment_features(
    valhalla_result: dict,
    traj_id: str,
    original_times: List[pd.Timestamp],
    mean_speed: float,
) -> pd.DataFrame:
    """
    Extracts segment-level features from Valhalla response.
    """
    segments = []
    edges = valhalla_result.get("edges", [])
    for i, edge in enumerate(edges):
        segments.append(
            {
                "traj_id": traj_id,
                "timestamp": original_times[i] if i < len(original_times) else None,
                "edge_id": edge.get("id"),
                "speed_segment": edge.get("speed"),
                "length_segment": edge.get("length"),
                "speed_limit": edge.get("speed_limit", 0),
                "base_speed": mean_speed,
            }
        )
    return pd.DataFrame(segments)


async def send_valhalla_request_async(
    payload: dict, session: aiohttp.ClientSession
) -> Optional[dict]:
    """
    Sends an asynchronous request to Valhalla for map matching.
    """
    try:
        async with session.post(
            "http://localhost:8002/trace_attributes", json=payload, timeout=30
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logging.warning(f"Valhalla response {resp.status}: {await resp.text()}")
    except Exception as e:
        logging.error(f"Aiohttp error: {e}")
    return None


async def process_partition_async(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a pandas partition asynchronously using Valhalla.
    """
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = []

        for traj_id, group in pdf.groupby("id"):
            group = group.sort_values("timestamp")
            times = group["timestamp"].tolist()
            mean_speed = group["speed"].mean()
            start = group["timestamp"].min()
            group["time"] = (group["timestamp"] - start).dt.total_seconds()

            shape = (
                group[["latitude", "longitude", "time"]]
                .rename(columns={"latitude": "lat", "longitude": "lon"})
                .to_dict(orient="records")
            )
            payload = {
                "shape": shape,
                "costing": "auto",
                "shape_match": "map_snap",
                "search_radius": 100,
                "trace_options": {"turn_penalty_factor": 500},
            }

            async def _task(traj_id, payload, times, mean_speed):
                async with sem:
                    result = await send_valhalla_request_async(payload, session)
                    if result:
                        return extract_segment_features(
                            result, traj_id, times, mean_speed
                        )
                    else:
                        return pd.DataFrame(columns=MAPPED_OUTPUT_SCHEMA.keys())

            tasks.append(_task(traj_id, payload, times, mean_speed))

        dfs = await asyncio.gather(*tasks)

    valid = [df.dropna(axis=1, how="all") for df in dfs if not df.empty]
    if valid:
        out = pd.concat(valid, ignore_index=True)
        return out.reindex(columns=list(MAPPED_OUTPUT_SCHEMA.keys()))
    else:
        empty = pd.DataFrame(columns=list(MAPPED_OUTPUT_SCHEMA.keys()))
        for c, dtype in MAPPED_OUTPUT_SCHEMA.items():
            if c != "timestamp":
                empty[c] = empty[c].astype(dtype)
        return empty


def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the async map matching on a pandas partition.
    """
    return asyncio.run(process_partition_async(pdf))


def map_matching():
    """
    Map matching pipeline using Dask and asynchronous Valhalla requests.
    """
    start_ts = time.time()
    logging.info("Starting map matching pipeline...")

    if os.path.exists(MAPPED_DF):
        logging.info("Output already exists, aborting.")
        return

    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="auto")

    with Client(cluster):
        ddf = dd.read_parquet(
            FILTERED_DF, engine="pyarrow", columns=CLEAN_REQUIRED_COLUMNS
        )
        matched = ddf.map_partitions(process_partition, meta=MAPPED_OUTPUT_SCHEMA)
        ensure_directory_exists(os.path.dirname(MAPPED_DF))

        with ProgressBar():
            matched.to_parquet(
                MAPPED_DF, engine="pyarrow", write_index=False, compression="snappy"
            )

        total = ddf["id"].nunique().compute()
        success = matched["traj_id"].nunique().compute()
        logging.info(f"Total: {total}, Success: {success}, Failure: {total - success}")
        logging.info(f"Total time: {time.time() - start_ts:.2f}s")
