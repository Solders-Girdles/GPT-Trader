"""
State Repositories for Tier-Specific Storage

Encapsulates tier-specific storage operations with a shared interface
for Redis (HOT), PostgreSQL (WARM), and S3 (COLD) tiers.
"""

from typing import Any, Protocol

from .postgres_repository import PostgresStateRepository

# Import tier-specific repositories from dedicated modules
from .redis_repository import RedisStateRepository
from .s3_repository import S3StateRepository


class StateRepository(Protocol):
    """
    Protocol defining the interface for tier-specific state repositories.

    All repositories must implement these methods for consistent access.
    """

    async def fetch(self, key: str) -> Any | None:
        """Fetch state value by key."""
        ...

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """Store state value with metadata."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete state by key."""
        ...

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        ...

    async def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        ...

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items at once.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        ...

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys at once.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        ...


__all__ = [
    "StateRepository",
    "RedisStateRepository",
    "PostgresStateRepository",
    "S3StateRepository",
]
