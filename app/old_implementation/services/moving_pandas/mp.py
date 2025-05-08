from typing import Dict, List

import folium
import geopandas as gpd
import movingpandas as mpd
import pandas as pd
from shapely.geometry import LineString

from app.db.db_utils import load_matched_probe_data, load_raw_probe_data
from app.schemas.location import LocationSTG
from app.services.moving_pandas.calcutations import (
    calculate_avg_speed,
    calculate_avg_speed_by_address,
    calculate_duration,
    calculate_probes_per_second,
)
from app.services.moving_pandas.parsing import (
    parse_csv_for_vehicle,
    process_matched_trajectory,
)


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


def create_trajectory_map_matched_address(
    collection: mpd.TrajectoryCollection, map_center: list
) -> folium.Map:
    """Cria um mapa com trajetórias MAP MATCHED, coloridas por street name, com info adicional do speed limit."""
    m = folium.Map(location=map_center, zoom_start=14)

    predefined_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "yellow",
        "brown",
        "pink",
        "gray",
        "black",
        "lightblue",
        "lightgreen",
        "lightred",
        "lightgray",
        "darkblue",
        "darkgreen",
        "darkred",
        "darkpurple",
        "beige",
        "cadetblue",
    ]
    street_colors: Dict[str, str] = {}
    color_index = 0

    for traj in collection.trajectories:
        if len(traj.df) < 2:
            continue

        speeds_by_address = calculate_avg_speed_by_address(traj)

        for address, speed in speeds_by_address.items():
            traj_subset = traj.df[traj.df["address"] == address]
            if len(traj_subset) < 2:
                continue

            if address not in street_colors:
                if color_index < len(predefined_colors):
                    street_colors[address] = predefined_colors[color_index]
                    color_index += 1
                else:
                    street_colors[address] = "gray"
            color = street_colors[address]

            # Recupera a informação de speed limit se presente na coluna "speed_limit"
            speed_limit = (
                traj_subset["speed_limit"].iloc[0]
                if "speed_limit" in traj_subset.columns
                else "N/A"
            )

            tooltip_text = (
                f"Matched Segment {traj.id} ({address})<br>"
                f"Start: {traj_subset.index[0]}<br>"
                f"End: {traj_subset.index[-1]}<br>"
                f"Avg Speed: {speed:.2f} km/h<br>"
                f"Speed Limit: {speed_limit}<br>"
                f"Duration: {(traj_subset.index[-1] - traj_subset.index[0]).total_seconds()/60:.2f} min<br>"
                f"Num Points: {len(traj_subset)}"
            )

            # Cria a linha com as coordenadas convertendo (x, y) para (lat, lng)
            line = LineString(traj_subset.geometry.tolist())
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in line.coords],
                color=color,
                weight=8,
                tooltip=tooltip_text,
            ).add_to(m)

            # Marcadores de início e fim da trajetória
            start_point, end_point = line.coords[0], line.coords[-1]
            folium.Marker(
                location=[start_point[1], start_point[0]],
                popup=f"Início ({address})",
                icon=folium.Icon(color=color, icon="play", prefix="fa"),
            ).add_to(m)
            folium.Marker(
                location=[end_point[1], end_point[0]],
                popup=f"Fim ({address})",
                icon=folium.Icon(color=color, icon="stop", prefix="fa"),
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

    map_filename = "/Users/gilsilva/Work/thesis/output/trajectories_raw.html"
    m.save(map_filename)
    print(f"Raw trajectory map saved as {map_filename}")


def test_matched_address_sumo(identifier, batch_size):
    """
    Testa a visualização de trajetórias com dados MAP MATCHED.
    """

    results = parse_csv_for_vehicle(
        "/Users/gilsilva/Work/thesis/input/sumo/fcd.csv", identifier
    )

    matched_results = process_matched_trajectory(identifier, results, batch_size)

    df = create_raw_df(matched_results)
    gdf = create_geodataframe(df)
    gdf = gdf.set_index("timestamp")

    collection = mpd.TrajectoryCollection(gdf, "identifier")

    map_center = [
        gdf.iloc[0].geometry.y,
        gdf.iloc[0].geometry.x,
    ]

    m = create_trajectory_map_matched_address(collection, map_center)

    map_filename = f"/Users/gilsilva/Work/thesis/output/trajectories_matched_address_sumo_{identifier}.html"
    m.save(map_filename)
    print(f"Matched trajectory map saved as {map_filename}")


def test_matched_address(session, identifier, batch_size):
    """
    Testa a visualização de trajetórias com dados MAP MATCHED e Snapped.
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

    m = create_trajectory_map_matched_address(collection, map_center)

    map_filename = (
        "/Users/gilsilva/Work/thesis/output/trajectories_matched_address.html"
    )
    m.save(map_filename)
    print(f"Matched trajectory map saved as {map_filename}")


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

    map_filename = "/Users/gilsilva/Work/thesis/output/trajectories_matched.html"
    m.save(map_filename)
    print(f"Matched trajectory map saved as {map_filename}")
