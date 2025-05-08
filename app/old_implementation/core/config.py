import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    api_prod_url: str
    api_prod_access_token: str
    api_stg_url: str
    api_stg_access_token: str

    class Config:
        env_file = ".env"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Retorna a URI de conexão do banco de dados para SQLAlchemy."""
        port_str = f":{self.postgres_port}" if self.postgres_port else ""
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}{port_str}/{self.postgres_db or ''}"


settings = Settings()
