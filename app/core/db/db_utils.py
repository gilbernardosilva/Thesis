from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.core.db.session import get_db
from app.models.probes import ProbeModel


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
            COPY probes(identifier, routing_mode, latitude, longitude, altitude, 
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
