import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    api_v1_str: Optional[str] = os.getenv("API_V1_STR")
    postgres_user: Optional[str] = os.getenv("POSTGRES_USER")
    postgres_password: Optional[str] = os.getenv("POSTGRES_PASSWORD")
    postgres_db: Optional[str] = os.getenv("POSTGRES_DB")
    postgres_host: Optional[str] = os.getenv("POSTGRES_HOST")
    postgres_port: Optional[int] = os.getenv("POSTGRES_PORT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Retorna a URI de conexão do banco de dados para SQLAlchemy."""
        port_str = f":{self.postgres_port}" if self.postgres_port else ""
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}{port_str}/{self.postgres_db or ''}"


settings = Settings()
