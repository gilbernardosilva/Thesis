from sqlalchemy import text
from sqlalchemy.orm import Session


def filter_probes_within_radius(
    db: Session,
    center_coordinates: tuple[float, float],
    probes_list: list[tuple[float, float]],
    radius: float,
) -> list[tuple[float, float]]:
    query = text(
        """
        SELECT lat, lon
        FROM (VALUES :values) AS v(lat, lon)
        WHERE ST_DWithin(
            ST_SetSRID(ST_MakePoint(v.lon, v.lat), 4326),
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :radius
        )
    """
    )

    result = db.execute(
        query,
        {
            "values": probes_list,
            "lon": center_coordinates[0],
            "lat": center_coordinates[1],
            "radius": radius,
        },
    )
    return result.fetchall()


from sqlalchemy import text
from sqlalchemy.orm import Session


def filter_probes_within_radius_and_time(
    db: Session,
    center: tuple[float, float, int],
    probes_list: list[tuple[float, float, int]],
    radius: float,
    time_deviation: int,
) -> list[tuple[float, float, int]]:
    query = text(
        """
        SELECT lat, lon, timestamp
        FROM (VALUES :values) AS v(lat, lon, timestamp)
        WHERE ST_DWithin(
            ST_SetSRID(ST_MakePoint(v.lon, v.lat), 4326),
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :radius
        )
        AND v.timestamp BETWEEN :timestamp_center - :time_deviation AND :timestamp_center + :time_deviation
    """
    )

    result = db.execute(
        query,
        {
            "values": probes_list,
            "lon": center[0],
            "lat": center[1],
            "timestamp_center": center[2],
            "radius": radius,
            "time_deviation": time_deviation,
        },
    )

    return result.fetchall()


def calculate_direction(
    db: Session, coordinates: list[tuple[float, float]]
) -> float | None:
    if len(coordinates) != 2:
        return None

    lat1, lon1 = coordinates[0]
    lat2, lon2 = coordinates[1]

    query = text(
        """
        SELECT degrees(ST_Azimuth(
            ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326),
            ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326)
        ))
    """
    )

    try:
        result = db.execute(
            query, {"lon1": lon1, "lat1": lat1, "lon2": lon2, "lat2": lat2}
        )
        return result.scalar()
    except Exception as e:
        print(f"Error in calculate_direction: {e}")
        return None


def calculate_velocity(
    db: Session, coordinates: list[tuple[float, float, float]]
) -> float | None:
    if len(coordinates) != 2:
        return None

    lat1, lon1, t1 = coordinates[0]
    lat2, lon2, t2 = coordinates[1]

    if t2 == t1:
        return 0

    query = text(
        """
        SELECT ST_Distance(
            geography(ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326)),
            geography(ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326))
        ) / (:t2 - :t1)
    """
    )

    try:
        result = db.execute(
            query,
            {
                "lon1": lon1,
                "lat1": lat1,
                "lon2": lon2,
                "lat2": lat2,
                "t1": t1,
                "t2": t2,
            },
        )
        return result.scalar()
    except Exception as e:
        print(f"Error in calculate_velocity: {e}")
        return None
