from typing import List, Tuple

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.models.probes import ProbeModel
from app.schemas.location import LocationSTG


def delete_table(db: Session, table_name: str) -> None:
    try:
        db.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


def get_table_names(db: Session) -> list[str]:
    inspector = inspect(db.connection())
    return inspector.get_table_names()


def import_probes_from_file(db: Session, file_path: str) -> None:
    try:
        with open(file_path, "r") as f:
            copy_sql = """
            COPY probes(identifier, routing_mode, lat, lng, altitude, 
                        course_over_ground, speed, hdop, timestamp, rt)
            FROM STDIN WITH (FORMAT csv, DELIMITER ';', HEADER false);
            """
            connection = db.connection()
            raw_connection = connection.connection
            cursor = raw_connection.cursor()
            cursor.copy_expert(copy_sql, f)
            raw_connection.commit()
    except Exception as e:
        db.rollback()
        raise e


def print_probes(db: Session) -> None:
    for probe in db.query(ProbeModel).limit(40).all():
        print(probe)


def get_probe_by_identifier_and_timestamp(
    db: Session, identifier: str, timestamp: str
) -> ProbeModel:
    probe = (
        db.query(ProbeModel)
        .filter(ProbeModel.identifier == identifier, ProbeModel.timestamp == timestamp)
        .first()
    )
    return probe


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
