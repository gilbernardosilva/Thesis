import csv
from typing import List

from app.schemas.location import LocationSTG, LocationSUMO
from app.services.api.api import fetch_off_route_info

DEFAULT_TIMESTAMP_OFFSET = 1733333333


def group_locations(
    locations: List[LocationSTG], batch_size: int = 5
) -> List[List[LocationSTG]]:
    batches = []

    for i in range(0, len(locations), batch_size):
        batch = locations[i : i + batch_size]

        if batch:
            batches.append(batch)

    return batches


def parse_csv_for_vehicle(
    csv_file_path: str, target_vehicle_id: str
) -> List[LocationSUMO]:
    locations = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            if row["vehicle_id"] == target_vehicle_id:
                try:
                    timestep = float(row["timestep_time"])
                    timestamp = int(timestep) + DEFAULT_TIMESTAMP_OFFSET
                    lat = float(row["vehicle_y"])
                    lng = float(row["vehicle_x"])
                    speed = float(row["vehicle_speed"])
                    course_over_ground = float(row["vehicle_angle"])
                except (KeyError, ValueError) as e:
                    continue

                location = LocationSUMO(
                    lat=lat,
                    lng=lng,
                    timestamp=timestamp,
                    speed=speed,
                    course_over_ground=course_over_ground,
                )
                locations.append(location)
    return locations


def process_matched_trajectory(
    identifier: str,
    locations: List[LocationSTG],
    batch_size: int = 5,
) -> List[dict]:
    """
    Processa os pontos de localização e retorna uma lista de dicionários com
    informações de map matching, incluindo endereço e speed limit.
    """
    batches = group_locations(locations, batch_size)
    matched_results = []

    for group in batches:
        result = fetch_off_route_info(group)
        if result and result.get("status") == "OK":
            snapped_road = result.get("snapped_road", {})
            street_address = snapped_road.get("street_name")
            group_aux = group
            while not street_address and len(group_aux) > 0:
                result = fetch_off_route_info(group_aux)
                if result and result.get("status") == "OK":
                    snapped_road = result.get("snapped_road", {})
                    street_address = snapped_road.get("street_name")

                if not street_address:
                    group_aux = group_aux[:-1]

            snap_point = snapped_road.get("snap_point", {})
            speed_info = result.get("speed_limit", {})
            travel_speed = speed_info.get("travel_speed")
            speed_limit_value = speed_info.get("speed_limit_value")

            matched_results.append(
                {
                    "lat": snap_point.get("lat"),
                    "lng": snap_point.get("lng"),
                    "travel_speed": travel_speed,
                    "timestamp": group[0].timestamp,
                    "num_points": len(group),
                    "identifier": identifier,
                    "address": street_address,
                    "speed_limit": speed_limit_value,
                }
            )
        else:
            print("API returned an error for group:", group)
    return matched_results
