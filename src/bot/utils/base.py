"""
Base classes and common patterns for GPT-Trader.

This module provides base classes that implement common patterns
used throughout the codebase to reduce duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel, ABC):
    """Base configuration class with common save/load functionality."""

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path where to save the configuration
        """
        from bot.utils.config import ConfigManager

        ConfigManager.save_json_config(self.dict(), path)

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            Configuration instance
        """
        from bot.utils.config import ConfigManager

        data = ConfigManager.load_json_config(path)
        return cls(**data)

    def merge(self: T, other: dict[str, Any] | T) -> T:
        """Merge this configuration with another.

        Args:
            other: Dictionary or configuration instance to merge

        Returns:
            New configuration instance with merged values
        """
        from bot.utils.config import ConfigManager

        base_dict = self.dict()
        other_dict = other.dict() if isinstance(other, BaseConfig) else other

        merged_dict = ConfigManager.merge_configs(base_dict, other_dict)
        return self.__class__(**merged_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path where to save the configuration
        """
        from bot.utils.config import ConfigManager

        ConfigManager.save_yaml_config(self.dict(), path)

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration instance
        """
        from bot.utils.config import ConfigManager

        data = ConfigManager.load_yaml_config(path)
        return cls(**data)


class BaseValidator(ABC):
    """Base validator class with common validation patterns."""

    @abstractmethod
    def validate(self, value: Any) -> Any:
        """Validate a value.

        Args:
            value: Value to validate

        Returns:
            Validated value
        """
        pass

    def validate_required(self, value: Any, field_name: str) -> Any:
        """Validate required field.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If value is None or empty
        """
        if value is None:
            raise ValueError(f"{field_name} is required")

        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

        return self.validate(value)

    def validate_optional(self, value: Any, default: Any = None) -> Any:
        """Validate optional field.

        Args:
            value: Value to validate (can be None)
            default: Default value if None

        Returns:
            Validated value or default
        """
        if value is None:
            return default

        return self.validate(value)


class BaseManager(ABC):
    """Base manager class with common lifecycle patterns."""

    def __init__(self) -> None:
        self._initialized = False
        self._running = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the manager."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the manager."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the manager."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    async def __aenter__(self) -> BaseManager:
        """Async context manager entry."""
        await self.initialize()
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
        await self.cleanup()


class BaseMetrics:
    """Base class for metrics collection."""

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}

    def record_metric(self, name: str, value: Any, tags: dict[str, str] | None = None) -> None:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        self._metrics[name] = {
            "value": value,
            "tags": tags or {},
            "timestamp": self._get_timestamp(),
        }

    def get_metric(self, name: str) -> Any | None:
        """Get metric value.

        Args:
            name: Metric name

        Returns:
            Metric value or None if not found
        """
        return self._metrics.get(name, {}).get("value")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()


class SingletonMeta(type):
    """Metaclass for implementing singleton pattern."""

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseCache:
    """Base caching class with TTL support."""

    def __init__(self, default_ttl: int = 3600) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        import time

        self._cache[key] = {"value": value, "expires_at": time.time() + (ttl or self._default_ttl)}

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed, False otherwise
        """
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if self._is_expired(entry)]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        import time

        return time.time() > entry["expires_at"]


class ContextManager:
    """Base context manager for resource management."""

    def __init__(self, resource_name: str = "resource") -> None:
        self.resource_name = resource_name
        self._entered = False

    def __enter__(self) -> ContextManager:
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._entered = False
        return False

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self._entered
