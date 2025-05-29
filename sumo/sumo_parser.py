import logging
import os
import time
import uuid

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster
from lxml import etree

from app.fcd.config.schemas import FILTERED_DATA_SCHEMA
from app.fcd.config.variables import (
    DASK_MEMORY_LIMIT,
    DASK_NUM_CORES,
    DASK_NUM_THREADS,
    SUMO_FCD_FILE,
    SUMO_FCD_PARSED_FILE,
    TS_END,
    TS_START,
)

TS_START = pd.to_datetime(TS_START)
TS_END = pd.to_datetime(TS_END)


def parse_fcd_xml(fcd_file: str, sim_start: pd.Timestamp) -> list:
    """Parse SUMO FCD XML file to extract vehicle trajectory data."""
    data = []
    parser = etree.iterparse(fcd_file, events=("end",), tag="timestep", huge_tree=True)
    for event, elem in parser:
        timestep = float(elem.get("time"))
        timestamp = sim_start + pd.Timedelta(seconds=timestep)
        if timestamp < TS_START or timestamp > TS_END:
            continue
        for vehicle in elem.findall("vehicle"):
            try:
                speed = float(vehicle.get("speed")) * 3.6
                lane = vehicle.get("lane")
                longitude = float(vehicle.get("x"))
                latitude = float(vehicle.get("y"))
                if not lane or not all(
                    [vehicle.get("speed"), vehicle.get("x"), vehicle.get("y")]
                ):
                    continue
                traj_id = str(
                    uuid.uuid5(
                        uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
                        vehicle.get("id"),
                    )
                )
                data.append((traj_id, timestamp, latitude, longitude, speed))
            except (ValueError, AttributeError) as e:
                logging.warning(f"Skipping invalid vehicle data: {e}")
        elem.clear()
    return data


def create_fcd_dataframe(data: list) -> pd.DataFrame:
    """Create a DataFrame from parsed FCD data."""
    if not data:
        logging.warning("No valid vehicle data to create DataFrame")
        return pd.DataFrame(
            columns=["traj_id", "timestamp", "latitude", "longitude", "speed"]
        )
    df = pd.DataFrame(
        data,
        columns=["traj_id", "timestamp", "latitude", "longitude", "speed"],
    )
    return df


def process_fcd(
    fcd_file: str = SUMO_FCD_FILE,
    sim_start: pd.Timestamp | None = None,
    output_parquet: str = SUMO_FCD_PARSED_FILE,
) -> dd.DataFrame:
    """Process SUMO FCD file to extract vehicle trajectory data using Dask and save to Parquet."""
    start = time.time()

    if not isinstance(sim_start, pd.Timestamp):
        logging.error("sim_start must be a pandas Timestamp")
        raise ValueError("sim_start must be a pandas Timestamp")

    logging.info("Initializing Dask cluster for FCD parsing...")
    cluster = LocalCluster(
        n_workers=DASK_NUM_CORES,
        threads_per_worker=DASK_NUM_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
    )
    client = Client(cluster)
    try:
        fcd_file = os.path.abspath(fcd_file)
        if not os.path.exists(fcd_file):
            logging.error(f"FCD file does not exist: {fcd_file}")
            raise FileNotFoundError(f"FCD file does not exist: {fcd_file}")

        logging.info(f"Parsing FCD file: {fcd_file}")
        data = parse_fcd_xml(fcd_file, sim_start)
        if not data:
            logging.warning(f"No valid vehicle data found in {fcd_file}")
            raise ValueError(f"No valid vehicle data found in {fcd_file}")

        pdf = create_fcd_dataframe(data)

        ddf = dd.from_pandas(pdf, npartitions=4)
        ddf = ddf.sort_values(by=["traj_id", "timestamp"])
        if output_parquet:
            logging.info(f"Saving parsed FCD data to {output_parquet}")
            ddf.to_parquet(
                output_parquet,
                engine="pyarrow",
                schema=FILTERED_DATA_SCHEMA,
            )
            logging.info(f"Wrote FCD data to {output_parquet}")

        logging.info(f"FCD parsing completed in {time.time() - start:.2f} seconds")
        return ddf
    except Exception as e:
        logging.error(f"Error in FCD parsing: {e}")
        raise
    finally:
        client.close()
        cluster.close()
        logging.info("Dask client and cluster closed for FCD parsing.")
