"""Data models for runtime coordinator bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class BrokerBootstrapError(RuntimeError):
    """Raised when broker initialization fails."""


@dataclass
class BrokerBootstrapArtifacts:
    """Artifacts returned from broker bootstrap routines."""

    broker: object
    registry_updates: dict[str, Any]
    event_store: object | None = None
    products: Sequence[object] = ()


__all__ = ["BrokerBootstrapArtifacts", "BrokerBootstrapError"]
