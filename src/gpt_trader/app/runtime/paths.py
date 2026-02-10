"""Helpers for resolving runtime storage locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.app.runtime.settings import RuntimeSettingsSnapshot

if TYPE_CHECKING:
    from gpt_trader.app.config import BotConfig


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved directories backing bot runtime storage."""

    storage_dir: Path
    event_store_root: Path


def resolve_runtime_paths(
    *,
    config: BotConfig | RuntimeSettingsSnapshot,
    profile: Any,
    bot_name: str = "runtime_data",
) -> RuntimePaths:
    """Materialise storage+event-store roots for the provided profile."""

    profile_value = profile.value if hasattr(profile, "value") else str(profile)

    runtime_root = Path(getattr(config, "runtime_root", "."))
    storage_dir = runtime_root / f"{bot_name}/{profile_value}"
    storage_dir.mkdir(parents=True, exist_ok=True)

    override_root_str = config.event_store_root_override
    if override_root_str is not None:
        override_root = Path(override_root_str)
        event_store_root = override_root
        if bot_name not in set(event_store_root.parts):
            event_store_root = event_store_root / bot_name / profile_value
    else:
        event_store_root = storage_dir

    event_store_root.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(storage_dir=storage_dir, event_store_root=event_store_root)


__all__ = ["RuntimePaths", "resolve_runtime_paths"]
