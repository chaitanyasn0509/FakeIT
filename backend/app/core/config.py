"""Backend runtime settings loaded from environment variables."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven backend settings."""

    database_url: str = "sqlite:///./uncloud.db"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60
    storage_backend: str = "local"
    local_storage_root: Path = Path("./data/storage")
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    config_path: Path = Path("config/default.yaml")
    allowed_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def _warn_insecure_defaults(self) -> Settings:
        """Emit a warning when running with the placeholder secret key."""
        if self.secret_key == "change-me":
            warnings.warn(
                "SECRET_KEY is set to the insecure default 'change-me'. "
                "Set the SECRET_KEY environment variable to a strong random value.",
                stacklevel=2,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached backend settings."""
    return Settings()
