from app.core.db.session import engine
from app.models.base import Base


def create_tables() -> None:
    with engine.begin() as conn:
        Base.metadata.create_all(bind=conn)
