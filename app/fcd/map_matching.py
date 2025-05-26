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
from dask.distributed import Client, LocalCluster

from app.fcd.config.schemas import FILTERED_DATA_SCHEMA, MAPPED_DATA_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    DASK_MEMORY_LIMIT,
    DASK_NUM_CORES,
    DASK_NUM_THREADS,
    FILTERED_DATA_FILE,
    MAPPED_DATA_FILE,
)

CONCURRENCY_LIMIT = 10


def _create_segment(
    edge: dict, traj_id: str, index: int, times: List[pd.Timestamp], speeds: List[float]
) -> Optional[dict]:
    """Create a single segment dictionary for a given edge and index.

    This function constructs a dictionary representing a single segment of a trajectory, containing
    attributes like trajectory ID, timestamp, edge ID, speeds, length, number of lanes, and road class.
    It ensures the index is valid for the provided times and speeds lists.

    Args:
        edge (dict): The edge data from Valhalla, containing attributes like way_id, speed, and length.
        traj_id (str): The trajectory ID.
        index (int): The index of the segment within the trajectory.
        times (List[pd.Timestamp]): List of timestamps for the trajectory.
        speeds (List[float]): List of speeds for the trajectory.

    Returns:
        Optional[dict]: A dictionary with segment attributes if the index is valid, otherwise None.
    """
    if index >= len(speeds) or index >= len(times):
        return None
    return {
        "traj_id": traj_id,
        "timestamp": times[index],
        "edge_id": edge.get("way_id"),
        "speed_segment": edge.get("speed"),
        "length_segment": edge.get("length"),
        "probe_speed": speeds[index],
        "num_lanes": edge.get("lane_count", 0),
        "road_class": edge.get("road_class", "unknown"),
    }


def extract_segment_features(
    valhalla_result: dict,
    traj_id: str,
    original_times: List[pd.Timestamp],
    speeds: List[float],
) -> pd.DataFrame:
    """Extract segment attributes from a Valhalla map matching result.

    This function processes the Valhalla response to create a Pandas DataFrame containing segment
    attributes for a trajectory, such as edge IDs, speeds, and road metadata. It iterates over the
    edges in the result, creating segments for each valid index range, and ensures the output DataFrame
    conforms to the MAPPED_DATA_SCHEMA by casting columns to the correct types. If no segments are
    created, it returns an empty DataFrame with the correct schema.

    Args:
        valhalla_result (dict): The Valhalla map matching response containing edge data.
        traj_id (str): The trajectory ID.
        original_times (List[pd.Timestamp]): List of timestamps for the trajectory.
        speeds (List[float]): List of speeds for the trajectory.

    Returns:
        pd.DataFrame: A DataFrame containing segment attributes, conforming to MAPPED_DATA_SCHEMA.
    """
    segments = []
    edges = valhalla_result.get("edges", [])
    for edge in edges:
        begin_idx = edge.get("begin_shape_index", 0)
        end_idx = min(
            edge.get("end_shape_index", len(original_times)), len(original_times)
        )
        segments.extend(
            _create_segment(edge, traj_id, i, original_times, speeds)
            for i in range(begin_idx, end_idx)
            if _create_segment(edge, traj_id, i, original_times, speeds) is not None
        )

    df = pd.DataFrame(segments)
    if not df.empty:
        for field in MAPPED_DATA_SCHEMA:
            if field.name in df.columns:
                try:
                    df[field.name] = df[field.name].astype(
                        field.type.to_pandas_dtype(), errors="ignore"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to cast column {field.name} to {field.type}: {e}"
                    )
    else:
        empty_data = {
            field.name: pd.Series(dtype=field.type.to_pandas_dtype())
            for field in MAPPED_DATA_SCHEMA
        }
        df = pd.DataFrame(empty_data)

    return df


async def send_valhalla_request_async(
    payload: dict, session: aiohttp.ClientSession
) -> Optional[dict]:
    """Send an asynchronous HTTP request to the Valhalla service for map matching.

    This function sends a POST request to the Valhalla trace_attributes endpoint with the provided
    payload, which includes trajectory shape and map matching parameters. It handles timeouts and
    client errors, logging issues and returning None if the request fails.

    Args:
        payload (dict): The JSON payload containing trajectory data and map matching parameters.
        session (aiohttp.ClientSession): The HTTP session for making the request.

    Returns:
        Optional[dict]: The Valhalla response as a dictionary if successful, otherwise None.
    """
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
    """Process a Pandas DataFrame partition asynchronously for map matching.

    This function groups the input DataFrame by trajectory ID, sorts each group by timestamp, and
    sends asynchronous map matching requests to Valhalla for each trajectory. It uses a semaphore to
    limit concurrency, processes the results into segment DataFrames, and concatenates valid results.
    If no valid results are produced, it returns an empty DataFrame with the correct schema.

    Args:
        pdf (pd.DataFrame): The Pandas DataFrame partition to process.

    Returns:
        pd.DataFrame: A DataFrame containing map-matched segments, conforming to MAPPED_DATA_SCHEMA.
    """
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
                "filters": {
                    "attributes": [
                        "edge.way_id",
                        "edge.speed",
                        "edge.length",
                        "edge.lane_count",
                        "edge.road_class",
                        "edge.names",
                        "edge.begin_shape_index",
                        "edge.end_shape_index",
                    ],
                    "action": "include",
                },
            }

            async def _task(traj_id, payload, times, speeds):
                async with sem:
                    result = await send_valhalla_request_async(payload, session)
                    if result:
                        return extract_segment_features(result, traj_id, times, speeds)
                    else:
                        logging.warning(f"Map matching failed for traj_id {traj_id}")
                        empty_data = {
                            field.name: pd.Series(dtype=field.type.to_pandas_dtype())
                            for field in MAPPED_DATA_SCHEMA
                        }
                        return pd.DataFrame(empty_data)

            tasks.append(_task(traj_id, payload, times, speeds))

        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for df in dfs:
            if isinstance(df, pd.DataFrame) and not df.empty:
                valid.append(df.dropna(axis=1, how="all"))
            elif isinstance(df, Exception):
                logging.error(f"Task failed with error: {df}")

        if valid:
            try:
                out = pd.concat(valid, ignore_index=True)
                return out.reindex(columns=[field.name for field in MAPPED_DATA_SCHEMA])
            except Exception as e:
                logging.error(f"Error concatenating DataFrames: {e}")
                empty_data = {
                    field.name: pd.Series(dtype=field.type.to_pandas_dtype())
                    for field in MAPPED_DATA_SCHEMA
                }
                return pd.DataFrame(empty_data)
        else:
            empty_data = {
                field.name: pd.Series(dtype=field.type.to_pandas_dtype())
                for field in MAPPED_DATA_SCHEMA
            }
            return pd.DataFrame(empty_data)


def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    """Run asynchronous map matching on a Pandas DataFrame partition.

    This function serves as a synchronous wrapper around the asynchronous partition processing
    function, running the async map matching logic and handling any errors. If processing fails,
    it returns an empty DataFrame with the correct schema.

    Args:
        pdf (pd.DataFrame): The Pandas DataFrame partition to process.

    Returns:
        pd.DataFrame: A DataFrame containing map-matched segments, conforming to MAPPED_DATA_SCHEMA.
    """
    try:
        return asyncio.run(process_partition_async(pdf))
    except Exception as e:
        logging.error(f"Error processing partition: {e}")
        empty_data = {
            field.name: pd.Series(dtype=field.type.to_pandas_dtype())
            for field in MAPPED_DATA_SCHEMA
        }
        return pd.DataFrame(empty_data)


def map_matching():
    """Run the map matching pipeline using Dask and Valhalla.

    This function orchestrates the map matching process by loading filtered data, applying map
    matching to each partition using Valhalla, and saving the results to a Parquet file. It uses a
    Dask LocalCluster for parallel processing, logs the number of successfully matched trajectories,
    and skips the pipeline if the output file already exists. The Dask client and cluster are properly
    shut down after execution.

    Returns:
        None

    Raises:
        Exception: If loading data, processing partitions, or saving results fails.
    """
    start_ts = time.time()
    logging.info("Starting map matching pipeline...")
    if os.path.exists(MAPPED_DATA_FILE):
        logging.info(f"Output already exists at {MAPPED_DATA_FILE}, aborting.")
        return
    cluster = LocalCluster(
        n_workers=DASK_NUM_CORES,
        threads_per_worker=DASK_NUM_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
    )
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


if __name__ == "__main__":
    map_matching()
