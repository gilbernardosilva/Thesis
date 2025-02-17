from sqlalchemy.orm import Session

from app.db.session import engine
from app.models.base import Base
from app.models.road_type import RoadTypeModel


def create_tables() -> None:
    with engine.begin() as conn:
        Base.metadata.create_all(bind=conn)


def init_road_type_parameters(db: Session) -> None:
    road_types = [
        {"type": "Motorway", "radius": 300, "time_window": 600},
        {"type": "Primary", "radius": 200, "time_window": 300},
        {"type": "Secondary", "radius": 150, "time_window": 240},
        {"type": "Tertiary", "radius": 100, "time_window": 180},
        {"type": "Residential", "radius": 80, "time_window": 120},
        {"type": "Service Road", "radius": 50, "time_window": 60},
    ]

    for road in road_types:
        exists = db.query(RoadTypeModel).filter_by(type=road["type"]).first()
        if not exists:
            db.add(RoadTypeModel(**road))

    db.commit()
