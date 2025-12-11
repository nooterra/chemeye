"""
Configuration management using Pydantic BaseSettings.
All config is loaded from environment variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: Literal["dev", "staging", "prod"] = "dev"

    # Database
    database_url: str = "sqlite:////data/chemeye.db"

    # NASA EarthData credentials
    nasa_earthdata_username: str = ""
    nasa_earthdata_password: str = ""

    # API Security
    secret_key: str = "dev-secret-change-in-production"
    admin_token: str = "admin-token-change-in-production"

    # API Settings
    api_v1_prefix: str = "/v1"
    cors_origins: list[str] = ["*"]

    # Logging
    log_level: str = "INFO"

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def data_dir(self) -> Path:
        """Directory for data storage (SQLite, cache, etc.)."""
        path = Path("./data")
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
