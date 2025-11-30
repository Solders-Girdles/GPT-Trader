"""Strategy registry for managing strategy profiles.

Provides:
- StrategyRegistry: Central registry for strategy discovery and management
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_trader.features.strategy_dev.config.strategy_profile import StrategyProfile

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Entry in the strategy registry."""

    profile: StrategyProfile
    registered_at: datetime = field(default_factory=datetime.now)
    file_path: Path | None = None
    is_active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRegistry:
    """Central registry for strategy profiles.

    Features:
    - Register and discover strategies
    - Version tracking
    - Active strategy management
    - Persistence and loading
    """

    storage_path: Path | None = None
    _entries: dict[str, RegistryEntry] = field(default_factory=dict)
    _active_strategy: str | None = None
    _callbacks: dict[str, list[Callable]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize registry from storage."""
        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from storage."""
        if not self.storage_path:
            return

        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)

                self._active_strategy = data.get("active_strategy")

                for entry_data in data.get("entries", []):
                    profile = StrategyProfile.from_dict(entry_data["profile"])
                    file_path = entry_data.get("file_path")

                    entry = RegistryEntry(
                        profile=profile,
                        registered_at=datetime.fromisoformat(entry_data["registered_at"]),
                        file_path=Path(file_path) if file_path else None,
                        is_active=entry_data.get("is_active", False),
                        metadata=entry_data.get("metadata", {}),
                    )
                    self._entries[profile.name] = entry

                logger.info(f"Loaded {len(self._entries)} strategies from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to storage."""
        if not self.storage_path:
            return

        registry_file = self.storage_path / "registry.json"

        entries_data = []
        for entry in self._entries.values():
            entries_data.append(
                {
                    "profile": entry.profile.to_dict(),
                    "registered_at": entry.registered_at.isoformat(),
                    "file_path": str(entry.file_path) if entry.file_path else None,
                    "is_active": entry.is_active,
                    "metadata": entry.metadata,
                }
            )

        data = {
            "last_updated": datetime.now().isoformat(),
            "active_strategy": self._active_strategy,
            "entries": entries_data,
        }

        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for registry events.

        Events:
        - registered: Strategy registered
        - updated: Strategy updated
        - activated: Strategy activated
        - removed: Strategy removed
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in registry callback: {e}")

    def register(
        self,
        profile: StrategyProfile,
        file_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEntry:
        """Register a strategy profile.

        Args:
            profile: Strategy profile to register
            file_path: Optional file path for the profile
            metadata: Additional metadata

        Returns:
            Registry entry
        """
        # Validate profile
        errors = profile.validate()
        if errors:
            raise ValueError(f"Invalid profile: {errors}")

        # Check for existing entry
        if profile.name in self._entries:
            logger.info(f"Updating existing strategy: {profile.name}")

        entry = RegistryEntry(
            profile=profile,
            file_path=file_path,
            metadata=metadata or {},
        )

        self._entries[profile.name] = entry
        self._save_registry()

        self._emit("registered", entry)
        logger.info(f"Registered strategy: {profile.name} v{profile.version}")

        return entry

    def get(self, name: str) -> StrategyProfile | None:
        """Get a strategy profile by name.

        Args:
            name: Strategy name

        Returns:
            Strategy profile or None
        """
        entry = self._entries.get(name)
        return entry.profile if entry else None

    def get_entry(self, name: str) -> RegistryEntry | None:
        """Get full registry entry by name.

        Args:
            name: Strategy name

        Returns:
            Registry entry or None
        """
        return self._entries.get(name)

    def list_strategies(
        self,
        tags: list[str] | None = None,
        environment: str | None = None,
    ) -> list[StrategyProfile]:
        """List all registered strategies.

        Args:
            tags: Filter by tags
            environment: Filter by environment

        Returns:
            List of strategy profiles
        """
        result = []
        for entry in self._entries.values():
            profile = entry.profile

            # Tag filter
            if tags and not any(tag in profile.tags for tag in tags):
                continue

            # Environment filter
            if environment and profile.environment != environment:
                continue

            result.append(profile)

        return result

    def activate(self, name: str) -> StrategyProfile:
        """Set a strategy as the active strategy.

        Args:
            name: Strategy name to activate

        Returns:
            Activated profile
        """
        entry = self._entries.get(name)
        if not entry:
            raise KeyError(f"Strategy '{name}' not found")

        # Deactivate current
        if self._active_strategy and self._active_strategy in self._entries:
            self._entries[self._active_strategy].is_active = False

        # Activate new
        entry.is_active = True
        self._active_strategy = name
        self._save_registry()

        self._emit("activated", entry)
        logger.info(f"Activated strategy: {name}")

        return entry.profile

    def get_active(self) -> StrategyProfile | None:
        """Get the currently active strategy.

        Returns:
            Active strategy profile or None
        """
        if not self._active_strategy:
            return None
        return self.get(self._active_strategy)

    def remove(self, name: str) -> bool:
        """Remove a strategy from the registry.

        Args:
            name: Strategy name to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._entries:
            return False

        entry = self._entries.pop(name)

        if self._active_strategy == name:
            self._active_strategy = None

        self._save_registry()
        self._emit("removed", entry)

        logger.info(f"Removed strategy: {name}")
        return True

    def compare(self, name1: str, name2: str) -> dict[str, Any]:
        """Compare two strategies.

        Args:
            name1: First strategy name
            name2: Second strategy name

        Returns:
            Comparison data
        """
        profile1 = self.get(name1)
        profile2 = self.get(name2)

        if not profile1 or not profile2:
            raise KeyError("One or both strategies not found")

        comparison = {
            "strategies": [name1, name2],
            "differences": {},
        }

        # Compare key settings
        dict1 = profile1.to_dict()
        dict2 = profile2.to_dict()

        def find_differences(d1: dict, d2: dict, path: str = "") -> None:
            for key in set(d1.keys()) | set(d2.keys()):
                full_path = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if v1 != v2:
                    if isinstance(v1, dict) and isinstance(v2, dict):
                        find_differences(v1, v2, full_path)
                    else:
                        comparison["differences"][full_path] = {
                            name1: v1,
                            name2: v2,
                        }

        find_differences(dict1, dict2)
        return comparison

    def search(self, query: str) -> list[StrategyProfile]:
        """Search strategies by name or description.

        Args:
            query: Search query

        Returns:
            Matching strategies
        """
        query_lower = query.lower()
        results = []

        for entry in self._entries.values():
            profile = entry.profile
            if (
                query_lower in profile.name.lower()
                or query_lower in profile.description.lower()
                or any(query_lower in tag.lower() for tag in profile.tags)
            ):
                results.append(profile)

        return results

    def summary(self) -> dict[str, Any]:
        """Get registry summary.

        Returns:
            Summary data
        """
        by_environment = {}
        by_tag = {}

        for entry in self._entries.values():
            profile = entry.profile

            # Count by environment
            env = profile.environment
            by_environment[env] = by_environment.get(env, 0) + 1

            # Count by tag
            for tag in profile.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "total_strategies": len(self._entries),
            "active_strategy": self._active_strategy,
            "by_environment": by_environment,
            "by_tag": by_tag,
            "strategies": [
                {
                    "name": e.profile.name,
                    "version": e.profile.version,
                    "environment": e.profile.environment,
                    "is_active": e.is_active,
                }
                for e in self._entries.values()
            ],
        }

    def __len__(self) -> int:
        """Get number of registered strategies."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._entries

    def __iter__(self):
        """Iterate over strategy profiles."""
        return iter(e.profile for e in self._entries.values())
