"""Runtime state container for BotConfig instances."""

from __future__ import annotations

from typing import Any

from gpt_trader.config.runtime_settings import RuntimeSettings


class ConfigState:
    """Mutable state cached alongside BotConfig instances."""

    def __init__(self) -> None:
        self.runtime_settings: RuntimeSettings | None = None
        self.profile_value: str | None = None
        self.overrides_snapshot: dict[str, Any] = {}
        self.config_snapshot: dict[str, Any] | None = None


__all__ = ["ConfigState"]
