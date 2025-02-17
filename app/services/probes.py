from typing import List, Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.models.probes import ProbeModel
from app.models.road_type import RoadTypeModel
from app.schemas.location import Location
from app.services.api import fetch_road_info


def query_all_probes_by_identifier(probe_base: ProbeModel, db: Session) -> list:
    query = text(
        """
        SELECT latitude, longitude, timestamp, identifier
        FROM probes
        WHERE probes.identifier = :identifier
        ORDER BY probes.timestamp
        """
    )

    result = db.execute(
        query,
        {
            "identifier": probe_base.identifier,
        },
    )

    probes = result.fetchall()

    return probes


def compute_direction(
    db: Session, lon1: float, lat1: float, lon2: float, lat2: float
) -> Optional[float]:
    query = text(
        """
        SELECT CASE
            WHEN :lon1 IS NULL OR :lat1 IS NULL OR :lon2 IS NULL OR :lat2 IS NULL
            THEN NULL
            ELSE degrees(ST_Azimuth(
                ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326),
                ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326)
            ))
        END
        """
    )
    result = db.execute(
        query, {"lon1": lon1, "lat1": lat1, "lon2": lon2, "lat2": lat2}
    ).fetchone()
    return result[0] if result is not None else None


def get_segment_parameters(db: Session, road_type: str):
    result = (
        db.query(RoadTypeModel)
        .filter(func.lower(RoadTypeModel.road_type) == road_type.lower())
        .first()
    )

    if result:
        return result.radius_meters, result.time_window_seconds
    else:
        return 200, 200


def is_within_radius_and_time(
    db: Session,
    center: tuple[float, float, int],
    probe: tuple[float, float, int],
    radius: float,
    time_deviation: int,
) -> bool:
    lat1, lon1, t1 = center
    lat2, lon2, t2 = probe

    t1 = int(t1)
    t2 = int(t2)

    if t2 < (t1 - time_deviation) or t2 > (t1 + time_deviation):
        return False

    query = text(
        """
        SELECT ST_Distance(
            geography(ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326)),
            geography(ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326))
        )
        """
    )

    result = db.execute(
        query,
        {
            "lon1": lon1,
            "lat1": lat1,
            "lon2": lon2,
            "lat2": lat2,
        },
    )

    distance = result.scalar()
    return distance <= radius


def filter_probes_within_radius_and_time(
    db: Session,
    center: tuple[float, float, int],
    probes_list: list[ProbeModel],
    radius: float,
    time_deviation: int,
) -> list[ProbeModel]:
    probes_within_radius_and_time = []

    for probe in probes_list:
        lat, lon, timestamp = probe.latitude, probe.longitude, probe.timestamp
        if is_within_radius_and_time(
            db, center, (lat, lon, timestamp), radius, time_deviation
        ):
            probes_within_radius_and_time.append(probe)

    return probes_within_radius_and_time


def filter_probes_by_direction(
    db: Session,
    probe_base: ProbeModel,
    probes: List[ProbeModel],
    direction_threshold: float = 90.0,
) -> List[ProbeModel]:
    sorted_probes = sorted(probes, key=lambda p: int(p.timestamp))
    base_ts = int(probe_base.timestamp)

    base_idx = next(
        (i for i, p in enumerate(sorted_probes) if int(p.timestamp) == base_ts), None
    )
    if base_idx is None or len(sorted_probes) < 2:
        return []

    reference_idx = base_idx - 1 if base_idx > 0 else base_idx + 1
    reference_direction = compute_direction(
        db,
        sorted_probes[reference_idx].longitude,
        sorted_probes[reference_idx].latitude,
        probe_base.longitude,
        probe_base.latitude,
    )

    if reference_direction is None:
        return []

    filtered_probes = []
    previous_probe = probe_base

    for probe in sorted_probes[base_idx + 1 :]:
        direction = compute_direction(
            db,
            previous_probe.longitude,
            previous_probe.latitude,
            probe.longitude,
            probe.latitude,
        )
        if direction is None:
            continue
        if abs(direction - reference_direction) <= direction_threshold:
            filtered_probes.append(probe)
            previous_probe = probe

    return filtered_probes


def filter_probes_by_street_name(
    candidate_probes: List[ProbeModel], probe_base: ProbeModel
) -> List[ProbeModel]:
    locations = [Location(lat=probe_base.latitude, lng=probe_base.longitude)] + [
        Location(lat=p.latitude, lng=p.longitude) for p in candidate_probes
    ]
    response = fetch_road_info(locations)
    print(locations)
    print(response)
    snapped_roads = response.get("snapped_roads", [])
    if not snapped_roads:
        print("No snapped roads found.")
        return []
    base_street = snapped_roads[0].get("street_name", "").strip().lower()
    print(f"Base street: {base_street}")
    filtered_candidates = []
    for probe, road_info in zip(candidate_probes, snapped_roads[1:]):
        candidate_street = road_info.get("street_name", "").strip().lower()
        print(f"Candidate probe {probe.identifier} street: {candidate_street}")
        if candidate_street == base_street:
            filtered_candidates.append(probe)
    print(f"Probes after street name filter: {len(filtered_candidates)}")
    return filtered_candidates


def process_segment(
    db: Session,
    probe_base: ProbeModel,
) -> dict:
    print(
        f"Processing probe {probe_base.identifier} at ({probe_base.latitude}, {probe_base.longitude})"
    )

    location_probe_base = [Location(lat=probe_base.latitude, lng=probe_base.longitude)]
    details = fetch_road_info(location_probe_base)
    road_type = details.get("attributes", {}).get("physical_class", "Unknown")
    print(f"Road type identified: {road_type}")

    radius, time_window = get_segment_parameters(db, road_type)
    print(f"Segment parameters -> Radius: {radius}m, Time Window: {time_window}s")

    all_probes = query_all_probes_by_identifier(probe_base, db)
    print(f"Total probes retrieved: {len(all_probes)}")

    probes_segment = filter_probes_within_radius_and_time(
        db,
        (probe_base.latitude, probe_base.longitude, probe_base.timestamp),
        all_probes,
        radius,
        time_window,
    )
    print(f"Probes after radius and time filter: {len(probes_segment)}")

    probes_direction_filtered = filter_probes_by_direction(
        db, probe_base, probes_segment, 90
    )
    print(f"Probes after direction filter: {len(probes_direction_filtered)}")

    probes_street_filtered = filter_probes_by_street_name(
        probes_direction_filtered, probe_base
    )
    print(f"Probes after street name filter: {len(probes_street_filtered)}")

    return {
        "road_type": road_type,
        "radius": radius,
        "time_window": time_window,
        "probe_count": len(probes_street_filtered),
    }
