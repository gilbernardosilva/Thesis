from geoalchemy2 import Geometry
from sqlalchemy import Column, Float, Integer, String

from app.models.base import Base


class ProbeModel(Base):
    __tablename__ = "probes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identifier = Column(String(255), nullable=False)
    routing_mode = Column(String(50), nullable=False)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    altitude = Column(Float)
    course_over_ground = Column(Float)
    speed = Column(Float)
    hdop = Column(Float)
    timestamp = Column(Integer, nullable=False)
    rt = Column(String(50))

    def __repr__(self):
        return f"<ProbeData(id={self.id}, identifier={self.identifier}, timestamp={self.timestamp})>"
