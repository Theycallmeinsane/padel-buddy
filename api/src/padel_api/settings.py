"""Configuration from environment variables (or a local .env file).

Every variable is prefixed ``PADEL_``, e.g. ``PADEL_DATA_DIR=/srv/padel``.
Defaults run everything on one machine with SQLite and local files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PADEL_", env_file=".env", extra="ignore")

    data_dir: Path = Field(Path("data"), description="Uploaded videos and run folders live here.")
    database_url: str | None = Field(
        None,
        description="SQLAlchemy URL. Defaults to SQLite in data_dir; use "
        "postgresql+psycopg://... for Supabase or any Postgres.",
    )
    api_token: str | None = Field(
        None,
        description="If set, every request must send 'Authorization: Bearer <token>' "
        "(or ?token= for media links). Leave unset for local-only use.",
    )
    cors_origins: list[str] = Field(["http://localhost:3000"], description="Web app origins allowed to call the API.")
    max_upload_gb: float = Field(25.0, gt=0)
    worker_poll_s: float = Field(2.0, gt=0)
    stale_job_s: float = Field(600.0, gt=0, description="A running job silent this long is re-queued.")
    web_dir: Path | None = Field(
        None, description="A built web app (web/out) to serve at /, so one process gives the whole app."
    )
    anthropic_api_key: SecretStr | None = Field(
        None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "PADEL_ANTHROPIC_API_KEY"),
        description="Claude API key for the written coaching summary in match reports. Optional.",
    )

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'padel.db').resolve()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
