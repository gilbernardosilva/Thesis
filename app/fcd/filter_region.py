import logging
import os
import pickle
import time

import dask
import dask.dataframe as dd
import geopandas as gpd
import osmnx as ox
from dask.diagnostics import ProgressBar
from distributed import Client, LocalCluster

from app.fcd.config.schemas import CLEAN_REQUIRED_COLUMNS
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    CLEAN_DF,
    EDGES_DATA_FILE,
    FILTERED_DF,
    REGION_NAME,
    REGION_TYPE,
)


def fetch_region_polygon_from_osm() -> gpd.GeoSeries:
    """
    Fetches the polygon geometry for the configured region using OSM.
    """
    logging.info(f"Fetching region polygon from OSM: {REGION_NAME} ({REGION_TYPE})")
    query = REGION_NAME if REGION_TYPE == "city" else f"{REGION_NAME}, country"
    gdf = ox.geocode_to_gdf(query)
    return gdf.geometry.iloc[0]


def load_cached_polygon() -> gpd.GeoSeries | None:
    """
    Loads the cached region polygon from disk, if available.
    """
    if not os.path.exists(EDGES_DATA_FILE):
        return None
    try:
        with open(EDGES_DATA_FILE, "rb") as f:
            cached = pickle.load(f)
            polygon = cached.get("region_polygon")
            if polygon is not None:
                logging.info(f"Loaded cached polygon from {EDGES_DATA_FILE}")
                return polygon
    except Exception as e:
        logging.warning(f"Failed to read cached polygon: {e}")
    return None


def get_region_polygon() -> gpd.GeoSeries:
    """
    Returns the region polygon, fetching and caching if necessary.
    """
    polygon = load_cached_polygon()
    if polygon is not None:
        return polygon
    polygon = fetch_region_polygon_from_osm()
    ensure_directory_exists(os.path.dirname(EDGES_DATA_FILE))
    with open(EDGES_DATA_FILE, "wb") as f:
        pickle.dump({"region_polygon": polygon}, f)
    logging.info(f"Cached polygon to {EDGES_DATA_FILE}")
    return polygon


def spatial_filter_partition(pdf, polygon, lat_col="latitude", lon_col="longitude"):
    """
    Applies spatial filtering to a partition using GeoPandas.
    """
    gdf = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf[lon_col], pdf[lat_col]),
        crs="EPSG:4326",
    )
    return gdf[gdf.geometry.within(polygon)].drop(columns="geometry")


def load_filter_region() -> None:
    """
    Filters cleaned data to the region of interest and saves it.
    """
    start = time.time()

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="auto",
    )

    with Client(cluster):
        if os.path.exists(FILTERED_DF):
            logging.info(f"Filtered data already exists at {FILTERED_DF}. Skipping.")
            return

        logging.info(f"Loading cleaned data from {CLEAN_DF}")
        ddf = dd.read_parquet(
            CLEAN_DF, columns=CLEAN_REQUIRED_COLUMNS, engine="pyarrow"
        )

        polygon = get_region_polygon()
        minx, miny, maxx, maxy = polygon.bounds

        logging.info("Applying bounding box filter")
        ddf = ddf[
            (ddf.longitude >= minx)
            & (ddf.longitude <= maxx)
            & (ddf.latitude >= miny)
            & (ddf.latitude <= maxy)
        ]

        logging.info("Applying spatial filter using GeoPandas")
        filtered_ddf = ddf.map_partitions(spatial_filter_partition, polygon)

        with ProgressBar():
            total_rows, filtered_rows = dask.compute(
                ddf.shape[0], filtered_ddf.shape[0]
            )

        logging.info(f"Rows before filtering: {total_rows}")
        logging.info(f"Rows after spatial filtering: {filtered_rows}")
        logging.info(
            f"Total rows removed: {total_rows - filtered_rows} "
            f"({(1 - filtered_rows / total_rows) * 100:.2f}%)"
        )

        ensure_directory_exists(os.path.dirname(FILTERED_DF))
        with ProgressBar():
            filtered_ddf.to_parquet(FILTERED_DF, engine="pyarrow", write_index=False)

        logging.info(
            f"Filtered data saved to {FILTERED_DF} in {time.time() - start:.2f}s"
        )
