"""Base abstractions for configuration monitoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConfigurationMonitor(ABC):
    """Abstract base for configuration monitoring components."""

    @abstractmethod
    def check_changes(self) -> list[Any]:
        """Check for configuration changes, return drift events if found."""

    @abstractmethod
    def get_current_state(self) -> dict[str, Any]:
        """Get current state for monitoring."""

    @property
    @abstractmethod
    def monitor_name(self) -> str:
        """Component name for logging."""


__all__ = ["ConfigurationMonitor"]
