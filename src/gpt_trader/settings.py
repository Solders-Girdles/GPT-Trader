"""Typed configuration backed by environment variables."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field, SecretStr

try:  # pragma: no cover - exercised via import failure in slim environments
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # Graceful fallback when optional dependency is absent
    from typing import Any

    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        """Lightweight stand-in that mirrors the BaseSettings API we rely on."""

        model_config: ClassVar[dict[str, Any]] = {}

        def __init__(self, **data: Any) -> None:
            data.pop("_env_file", None)  # Ignore env-loading kwargs in fallback mode
            super().__init__(**data)

    SettingsConfigDict = dict[str, Any]

_DEFAULT_ENV_FILES: tuple[Path, ...] = (
    Path(".env"),
    Path("tradegpt/.env"),
    Path("config/environments/.env"),
)


class Settings(BaseSettings):
    """Application configuration loaded from the environment and optional `.env` files."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(..., description="API key used for OpenAI model access.")
    td_api_key: SecretStr | None = Field(
        default=None, description="TD Ameritrade client id (optional)."
    )
    td_redirect_uri: str | None = Field(
        default=None, description="OAuth redirect URI for TD Ameritrade flows."
    )
    td_token_path: str | None = Field(
        default=None, description="Filesystem path to persisted TD Ameritrade token."
    )
    account_id: str | None = Field(
        default=None, description="Broker account identifier for trading operations."
    )

    model: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model identifier.",
    )
    log_dir: Path = Field(
        default=Path("reports"),
        description="Directory where runtime reports and logs should be written.",
    )

    @property
    def openai_key(self) -> str:
        """Expose the OpenAI key as a plain string for call sites that require it."""
        return self.openai_api_key.get_secret_value()


def _existing_env_files() -> list[str]:
    return [str(path) for path in _DEFAULT_ENV_FILES if path.exists()]


@lru_cache
def get_settings(_env_files: Sequence[str] | None = None) -> Settings:
    """Load settings once per process, respecting `.env` fallbacks."""
    env_files = list(_env_files) if _env_files is not None else _existing_env_files()
    if env_files:
        return Settings(_env_file=env_files)
    return Settings()


__all__ = ["Settings", "get_settings"]
