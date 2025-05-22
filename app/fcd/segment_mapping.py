import asyncio
import logging
import os
import time
from typing import List, Optional

import aiohttp
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from distributed import Client, LocalCluster

from app.fcd.config.schemas import (
    FILTERED_DATA_SCHEMA,
    MAPPED_DATA_PANDAS_DTYPES,
    MAPPED_DATA_SCHEMA,
)
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import FILTERED_DATA_FILE, MAPPED_DATA_FILE

CONCURRENCY_LIMIT = 10


def extract_segment_features(
    valhalla_result: dict,
    traj_id: str,
    original_times: List[pd.Timestamp],
    speeds: List[float],
) -> pd.DataFrame:
    """Extracts segment-level features from Valhalla response, copying speed directly from filtered DataFrame."""
    segments = []
    edges = valhalla_result.get("edges", [])
    for i, edge in enumerate(edges):
        if i < len(original_times) and i < len(speeds):
            segments.append(
                {
                    "traj_id": traj_id,
                    "timestamp": original_times[i],
                    "edge_id": edge.get("id"),
                    "speed_segment": edge.get("speed"),
                    "length_segment": edge.get("length"),
                    "speed_limit": edge.get("speed_limit", 0.0),
                    "speed": speeds[i],
                }
            )
    df = pd.DataFrame(segments)
    if not df.empty:
        for col, dtype in MAPPED_DATA_PANDAS_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype, errors="ignore")
    return df


async def send_valhalla_request_async(
    payload: dict, session: aiohttp.ClientSession
) -> Optional[dict]:
    """Sends an asynchronous request to Valhalla for map matching."""
    try:
        async with session.post(
            "http://localhost:8002/trace_attributes", json=payload, timeout=30
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except aiohttp.ClientError as e:
        logging.error(f"Aiohttp client error: {e}")
    except asyncio.TimeoutError:
        logging.error("Valhalla request timed out")
    return None


async def process_partition_async(pdf: pd.DataFrame) -> pd.DataFrame:
    """Processes a pandas partition asynchronously using Valhalla."""
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for traj_id, group in pdf.groupby("id"):
            group = group.sort_values("timestamp")
            times = group["timestamp"].tolist()
            speeds = group["speed"].tolist()
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

            async def _task(traj_id, payload, times, speeds):
                async with sem:
                    result = await send_valhalla_request_async(payload, session)
                    if result:
                        return extract_segment_features(result, traj_id, times, speeds)
                    else:
                        logging.warning(f"Map matching failed for traj_id {traj_id}")
                        return pd.DataFrame(
                            columns=[field.name for field in MAPPED_DATA_SCHEMA]
                        )

            tasks.append(_task(traj_id, payload, times, speeds))
        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for df in dfs:
            if isinstance(df, pd.DataFrame) and not df.empty:
                valid.append(df.dropna(axis=1, how="all"))
            elif isinstance(df, Exception):
                logging.error(f"Task failed with error: {df}")
        if valid:
            out = pd.concat(valid, ignore_index=True)
            return out.reindex(columns=[field.name for field in MAPPED_DATA_SCHEMA])
        else:
            empty = pd.DataFrame(columns=[field.name for field in MAPPED_DATA_SCHEMA])
            for col, dtype in MAPPED_DATA_PANDAS_DTYPES.items():
                empty[col] = empty[col].astype(dtype, errors="ignore")
            return empty


def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs the async map matching on a pandas partition."""
    try:
        return asyncio.run(process_partition_async(pdf))
    except Exception as e:
        logging.error(f"Error processing partition: {e}")
        return pd.DataFrame(columns=[field.name for field in MAPPED_DATA_SCHEMA])


def map_matching():
    """Map matching pipeline using Dask and asynchronous Valhalla requests."""
    start_ts = time.time()
    logging.info("Starting map matching pipeline...")
    if os.path.exists(MAPPED_DATA_FILE):
        logging.info(f"Output already exists at {MAPPED_DATA_FILE}, aborting.")
        return
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
    client = Client(cluster)
    try:

        ddf = dd.read_parquet(
            FILTERED_DATA_FILE, engine="pyarrow", schema=FILTERED_DATA_SCHEMA
        )
        matched = ddf.map_partitions(process_partition)
        ensure_directory_exists(os.path.dirname(MAPPED_DATA_FILE))
        with ProgressBar():
            matched.to_parquet(
                MAPPED_DATA_FILE,
                engine="pyarrow",
                write_index=False,
                compression="snappy",
                schema=MAPPED_DATA_SCHEMA,
            )
        total = ddf["id"].nunique().compute()
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
