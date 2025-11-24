"""Helpers for resolving runtime storage locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpt_trader.config.runtime_settings import RuntimeSettings


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved directories backing bot runtime storage."""

    storage_dir: Path
    event_store_root: Path


def resolve_runtime_paths(
    *,
    settings: RuntimeSettings,
    profile: Any,
    bot_name: str = "coinbase_trader",
) -> RuntimePaths:
    """Materialise storage+event-store roots for the provided profile."""

    profile_value = profile.value if hasattr(profile, "value") else str(profile)

    storage_dir = settings.runtime_root / f"{bot_name}/{profile_value}"
    storage_dir.mkdir(parents=True, exist_ok=True)

    override_root = settings.event_store_root_override
    if override_root is not None:
        event_store_root = override_root
        if bot_name not in set(event_store_root.parts):
            event_store_root = event_store_root / bot_name / profile_value
    else:
        event_store_root = storage_dir

    event_store_root.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(storage_dir=storage_dir, event_store_root=event_store_root)


__all__ = ["RuntimePaths", "resolve_runtime_paths"]
