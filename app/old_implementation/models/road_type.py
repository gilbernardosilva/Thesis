from sqlalchemy import Column, Integer, String

from app.models.base import Base


class RoadTypeModel(Base):
    __tablename__ = "road_type_parameters"

    road_type = Column(String(50), primary_key=True)
    radius_meters = Column(Integer, nullable=False)
    time_window_seconds = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<RoadTypeParameters(road_type={self.road_type}, radius={self.radius_meters}, time_window={self.time_window_seconds})>"
