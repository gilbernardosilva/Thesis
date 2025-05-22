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

from app.fcd.config.schemas import CLEAN_DATA_SCHEMA, FILTERED_DATA_SCHEMA
from app.fcd.config.utils import ensure_directory_exists
from app.fcd.config.variables import (
    CLEAN_DATA_FILE,
    EDGES_DATA_FILE,
    FILTERED_DATA_FILE,
    REGION_NAME,
    REGION_TYPE,
)


def fetch_region_polygon_from_osm() -> gpd.GeoSeries:
    """
    Fetches the polygon geometry for the configured region using OSM.
    """
    logging.info(f"Fetching region polygon from OSM: {REGION_NAME} ({REGION_TYPE})")
    query = REGION_NAME if REGION_TYPE == "city" else f"{REGION_NAME}, country"
    try:
        gdf = ox.geocode_to_gdf(query)
        return gdf.geometry.iloc[0]
    except Exception as e:
        logging.error(f"Failed to fetch region polygon from OSM: {e}")
        raise


def load_cached_polygon() -> gpd.GeoSeries | None:
    """
    Loads the cached region polygon from disk, if available.
    """
    if not os.path.exists(EDGES_DATA_FILE):
        logging.info(f"No cached polygon found at {EDGES_DATA_FILE}")
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

    edge_dir = os.path.dirname(EDGES_DATA_FILE)
    logging.info(f"Ensuring directory exists: {edge_dir}")
    try:
        ensure_directory_exists(EDGES_DATA_FILE)
        if not os.path.exists(edge_dir):
            raise OSError(f"Failed to create directory {edge_dir}")
        logging.info(f"Directory {edge_dir} verified or created")
    except Exception as e:
        logging.error(f"Error creating directory {edge_dir}: {e}")
        raise

    try:
        polygon = fetch_region_polygon_from_osm()
        with open(EDGES_DATA_FILE, "wb") as f:
            pickle.dump({"region_polygon": polygon}, f)
        logging.info(f"Cached polygon to {EDGES_DATA_FILE}")
    except Exception as e:
        logging.error(f"Error caching polygon to {EDGES_DATA_FILE}: {e}")
        raise

    return polygon


def spatial_filter_partition(pdf, polygon, lat_col="latitude", lon_col="longitude"):
    """
    Applies spatial filtering to a partition using GeoPandas.
    """
    try:
        gdf = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf[lon_col], pdf[lat_col]),
            crs="EPSG:4326",
        )
        filtered_gdf = gdf[gdf.geometry.within(polygon)].drop(columns="geometry")
        return filtered_gdf
    except Exception as e:
        logging.error(f"Error in spatial filtering partition: {e}")
        return pdf


def load_filter_region() -> None:
    """
    Filters cleaned data to the region of interest and saves it.
    """
    start = time.time()

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="2.5GB",
    )
    client = Client(cluster)

    try:
        if os.path.exists(FILTERED_DATA_FILE):
            logging.info(
                f"Filtered data already exists at {FILTERED_DATA_FILE}. Skipping."
            )
            return

        logging.info(f"Loading cleaned data from {CLEAN_DATA_FILE}")
        try:
            ddf = dd.read_parquet(
                CLEAN_DATA_FILE, engine="pyarrow", schema=CLEAN_DATA_SCHEMA
            )
            logging.info(f"Loaded columns: {ddf.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error loading cleaned data from {CLEAN_DATA_FILE}: {e}")
            raise

        total_rows_before_filter = ddf.shape[0]

        try:
            polygon = get_region_polygon()
        except Exception as e:
            logging.error(f"Error obtaining region polygon: {e}")
            raise

        minx, miny, maxx, maxy = polygon.bounds
        logging.info(
            f"Bounding box coordinates: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
        )

        logging.info("Applying bounding box filter")
        ddf_bbox = ddf[
            (ddf.longitude >= minx)
            & (ddf.longitude <= maxx)
            & (ddf.latitude >= miny)
            & (ddf.latitude <= maxy)
        ]

        logging.info("Applying spatial filter using GeoPandas")
        filtered_ddf = ddf_bbox.map_partitions(spatial_filter_partition, polygon)

        # Compute total and filtered counts
        with ProgressBar():
            total_rows, filtered_rows = dask.compute(
                total_rows_before_filter, filtered_ddf.shape[0]
            )

        percent_kept = (filtered_rows / total_rows) * 100 if total_rows else 0
        percent_removed = 100 - percent_kept

        logging.info(f"Rows before filtering: {total_rows}")
        logging.info(f"Rows after filtering: {filtered_rows}")
        logging.info(
            f"Filtered out {total_rows - filtered_rows} rows "
            f"({percent_removed:.2f}%% removed, {percent_kept:.2f}%% kept)"
        )

        ensure_directory_exists(os.path.dirname(FILTERED_DATA_FILE))
        with ProgressBar():
            filtered_ddf.to_parquet(
                FILTERED_DATA_FILE,
                engine="pyarrow",
                schema=FILTERED_DATA_SCHEMA,
                write_index=False,
                compression="snappy",
            )

        logging.info(
            f"Filtered data saved to {FILTERED_DATA_FILE} in {time.time() - start:.2f}s"
        )

    finally:
        logging.info("Shutting down Dask client and cluster...")
        client.close()
        cluster.close()
