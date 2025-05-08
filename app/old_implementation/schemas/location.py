from pydantic import BaseModel


class Location(BaseModel):
    lat: float
    lng: float


class LocationSTG(BaseModel):
    lat: float
    lng: float
    altitude: float
    timestamp: int
    speed: float
    course_over_ground: float


class LocationSUMO(BaseModel):
    lat: float
    lng: float
    timestamp: int
    speed: float
    course_over_ground: float
