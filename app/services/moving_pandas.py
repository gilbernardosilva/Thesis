import random
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import folium
import geopandas as gpd
import movingpandas as mpd
import pandas as pd
from folium import plugins
from shapely.geometry import LineString
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.probes import ProbeModel
from app.schemas.location import LocationSTG
from app.services.api import fetch_off_route_info


def load_raw_probe_data_batch(session: Session) -> List[Tuple[float, float, int, str]]:

    subquery = session.query(ProbeModel.identifier).distinct().limit(5).subquery()

    results = (
        session.query(
            ProbeModel.lat, ProbeModel.lng, ProbeModel.timestamp, ProbeModel.identifier
        )
        .filter(ProbeModel.identifier.in_(subquery))
        .order_by(ProbeModel.identifier, ProbeModel.timestamp)
        .all()
    )

    return results


def load_raw_probe_data(session: Session, identifier: str) -> List[Tuple]:
    """Carrega dados de probes para um identificador específico."""

    results = (
        session.query(
            ProbeModel.lat, ProbeModel.lng, ProbeModel.timestamp, ProbeModel.identifier
        )
        .filter_by(identifier=identifier)
        .order_by(ProbeModel.timestamp)
        .all()
    )

    return results


def load_matched_probe_data(session: Session, identifier: str) -> List[LocationSTG]:
    probes = (
        session.query(ProbeModel)
        .filter_by(identifier=identifier)
        .order_by(ProbeModel.timestamp)
        .all()
    )
    location_data = [LocationSTG(**probe.__dict__) for probe in probes]
    return location_data


def create_raw_df(raw_results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_results)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def create_matched_df(matched_results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(matched_results)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Converte um DataFrame para GeoDataFrame usando latitude e longitude.
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lng, df.lat),
        crs="EPSG:4326",
    )
    print("GeoDataFrame created with geometry column.")
    return gdf


def calculate_avg_speed(traj: mpd.Trajectory) -> float:
    """
    Para dados RAW: calcula a velocidade média (em km/h) usando MovingPandas.
    """
    traj.add_speed(overwrite=True)
    avg_speed = traj.df.speed.mean() * 3.6
    return avg_speed


def calculate_duration(start_time, end_time) -> timedelta:
    duration = end_time - start_time
    return duration


def calculate_probes_per_second(df: pd.DataFrame) -> float:
    total_probes = len(df)
    time_duration = (df.index[-1] - df.index[0]).total_seconds()
    return total_probes / time_duration if time_duration > 0 else 0


def group_locations(
    locations: List[LocationSTG], batch_size: int = 5, overlap_size: int = 1
) -> List[List[LocationSTG]]:
    batches = []

    for i in range(0, len(locations), batch_size - overlap_size):
        batch = locations[i : i + batch_size]

        if batch:
            batches.append(batch)

    return batches


# def group_locations(
#     locations: List[LocationSTG], batch_size: int = 5
# ) -> List[List[LocationSTG]]:
#     batches = []

#     for i in range(0, len(locations), batch_size):
#         batch = locations[i : i + batch_size]

#         if batch:
#             batches.append(batch)

#     return batches


def process_matched_trajectory(
    identifier: str,
    locations: List[LocationSTG],
    batch_size: int = 5,
) -> List[dict]:
    batches = group_locations(locations, batch_size)
    matched_results = []
    for group in batches:
        result = fetch_off_route_info(group)
        if result and result.get("status") == "OK":
            snapped_road = result.get("snapped_road", {})
            street_address = snapped_road.get("street_address")
            snap_point = snapped_road.get("snap_point", {})
            travel_speed = result.get("speed_limit", {}).get("travel_speed", None)
            matched_results.append(
                {
                    "lat": snap_point.get("lat"),
                    "lng": snap_point.get("lng"),
                    "travel_speed": travel_speed,
                    "timestamp": group[0].timestamp,
                    "num_points": len(group),
                    "identifier": identifier,
                    "address": street_address,
                }
            )
        else:
            print("API returned an error for group:", group)
    return matched_results


def create_trajectory_map_raw(
    collection: mpd.TrajectoryCollection, map_center: list
) -> folium.Map:
    """
    Cria um mapa com trajetórias RAW (não map matched).
    """
    m = folium.Map(location=map_center, zoom_start=14)

    for traj in collection.trajectories:
        if len(traj.df) < 2:
            continue

        avg_speed = calculate_avg_speed(traj)
        start_time = traj.df.index[0]
        end_time = traj.df.index[-1]
        duration = calculate_duration(start_time, end_time)
        duration_minutes = duration.total_seconds() / 60
        probes_per_sec = calculate_probes_per_second(traj.df)

        tooltip_text = (
            f"Trajectory {traj.id}<br>Start: {start_time}<br>End: {end_time}<br>"
            f"Avg Speed: {avg_speed:.2f} km/h<br>Duration: {duration_minutes:.2f} min<br>"
            f"Probes/sec: {probes_per_sec:.2f}<br>Num Points: {len(traj.df)}"
        )

        line = LineString(traj.df.geometry.tolist())
        folium.PolyLine(
            locations=[(p[1], p[0]) for p in line.coords],
            color="red",
            weight=10,
            tooltip=tooltip_text,
        ).add_to(m)

        coords = list(line.coords)
        start_point = coords[0]
        end_point = coords[-1]

        folium.Marker(
            location=[start_point[1], start_point[0]],
            popup="Início",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            location=[end_point[1], end_point[0]],
            popup="Fim",
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        ).add_to(m)

    return m


def create_trajectory_map_matched(
    collection: mpd.TrajectoryCollection, map_center: list
) -> folium.Map:
    """
    Cria um mapa com trajetórias MAP MATCHED.
    Assume que os dados matchados possuem a coluna 'travel_speed' com a velocidade (já em km/h).
    """
    m = folium.Map(location=map_center, zoom_start=14)

    for traj in collection.trajectories:
        if len(traj.df) < 2:
            continue

        avg_speed = calculate_avg_speed(traj)
        start_time = traj.df.index[0]
        end_time = traj.df.index[-1]
        duration = calculate_duration(start_time, end_time)
        duration_minutes = duration.total_seconds() / 60
        probes_per_sec = calculate_probes_per_second(traj.df)

        tooltip_text = (
            f"Matched Segment {traj.id}<br>Start: {start_time}<br>End: {end_time}<br>"
            f"Avg Speed: {avg_speed:.2f} km/h<br>Duration: {duration_minutes:.2f} min<br>"
            f"Probes/sec: {probes_per_sec:.2f}<br>Num Points: {len(traj.df)}"
        )

        line = LineString(traj.df.geometry.tolist())
        folium.PolyLine(
            locations=[(p[1], p[0]) for p in line.coords],
            color="blue",
            weight=8,
            tooltip=tooltip_text,
        ).add_to(m)

        coords = list(line.coords)
        start_point = coords[0]
        end_point = coords[-1]

        folium.Marker(
            location=[start_point[1], start_point[0]],
            popup="Início",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            location=[end_point[1], end_point[0]],
            popup="Fim",
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        ).add_to(m)

    return m


def test_raw(session, identifier):
    """
    Testa a visualização de trajetórias com dados RAW (sem map matching).
    """

    results = load_raw_probe_data(session, identifier)
    df = create_raw_df(results)
    gdf = create_geodataframe(df)
    gdf = gdf.set_index("timestamp")

    collection = mpd.TrajectoryCollection(gdf, "identifier")

    map_center = [
        gdf.iloc[0].geometry.y,
        gdf.iloc[0].geometry.x,
    ]

    m = create_trajectory_map_raw(collection, map_center)

    map_filename = "trajectories_raw.html"
    m.save(map_filename)
    print(f"Raw trajectory map saved as {map_filename}")


def test_matched(session, identifier, batch_size):
    """
    Testa a visualização de trajetórias com dados MAP MATCHED.
    """

    results = load_matched_probe_data(session, identifier)
    matched_results = process_matched_trajectory(identifier, results, batch_size)

    df = create_raw_df(matched_results)
    gdf = create_geodataframe(df)
    gdf = gdf.set_index("timestamp")

    collection = mpd.TrajectoryCollection(gdf, "identifier")

    map_center = [
        gdf.iloc[0].geometry.y,
        gdf.iloc[0].geometry.x,
    ]

    m = create_trajectory_map_matched(collection, map_center)

    map_filename = "trajectories_matched.html"
    m.save(map_filename)
    print(f"Matched trajectory map saved as {map_filename}")
