"""Persistence layer for strategy artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpt_trader.config.path_registry import STRATEGY_ARTIFACT_DIR
from gpt_trader.utilities.logging_patterns import get_logger

from .models import StrategyArtifact

logger = get_logger(__name__, component="strategy_artifacts")


class StrategyArtifactError(RuntimeError):
    """Raised when a strategy artifact cannot be loaded or validated."""


@dataclass(frozen=True)
class ArtifactRegistry:
    """Registry mapping profiles to active artifact ids."""

    profiles: dict[str, str]
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": dict(self.profiles),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRegistry":
        return cls(
            profiles=dict(data.get("profiles") or {}),
            updated_at=data.get("updated_at"),
        )


class StrategyArtifactStore:
    """Read/write strategy artifacts and active registry."""

    def __init__(self, root: Path | None = None) -> None:
        env_root = os.getenv("STRATEGY_ARTIFACT_ROOT")
        self._root = Path(env_root) if env_root else (root or STRATEGY_ARTIFACT_DIR)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def artifact_path(self, artifact_id: str) -> Path:
        return self._root / f"{artifact_id}.json"

    def registry_path(self) -> Path:
        return self._root / "active.json"

    def save(self, artifact: StrategyArtifact) -> Path:
        path = self.artifact_path(artifact.artifact_id)
        path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))
        return path

    def load(self, artifact_id_or_path: str | Path) -> StrategyArtifact:
        path = Path(artifact_id_or_path)
        if path.suffix != ".json":
            path = self.artifact_path(str(artifact_id_or_path))
        if not path.exists():
            raise StrategyArtifactError(f"Artifact not found: {path}")
        data = json.loads(path.read_text())
        artifact = StrategyArtifact.from_dict(data)
        if not artifact.artifact_id:
            raise StrategyArtifactError(f"Invalid artifact file: {path}")
        return artifact

    def list_artifacts(self) -> list[StrategyArtifact]:
        artifacts: list[StrategyArtifact] = []
        for path in sorted(self._root.glob("*.json")):
            if path.name == "active.json":
                continue
            try:
                artifacts.append(self.load(path))
            except StrategyArtifactError as exc:
                logger.warning("Skipping invalid artifact", path=str(path), error=str(exc))
        return artifacts

    def publish(
        self,
        artifact_id: str,
        *,
        approved_by: str | None = None,
        notes: str | None = None,
    ) -> StrategyArtifact:
        artifact = self.load(artifact_id)
        artifact.approved = True
        artifact.approved_at = datetime.now(timezone.utc).isoformat()
        artifact.approved_by = approved_by
        if notes:
            artifact.notes = notes
        self.save(artifact)
        return artifact

    def set_active(self, profile: str, artifact_id: str) -> ArtifactRegistry:
        registry = self._load_registry()
        profiles = dict(registry.profiles)
        profiles[profile] = artifact_id
        updated_at = datetime.now(timezone.utc).isoformat()
        updated = ArtifactRegistry(profiles=profiles, updated_at=updated_at)
        self._save_registry(updated)
        return updated

    def resolve_active(self, profile: str) -> str | None:
        registry = self._load_registry()
        return registry.profiles.get(profile)

    def _load_registry(self) -> ArtifactRegistry:
        path = self.registry_path()
        if not path.exists():
            return ArtifactRegistry(profiles={})
        data = json.loads(path.read_text())
        return ArtifactRegistry.from_dict(data)

    def _save_registry(self, registry: ArtifactRegistry) -> None:
        path = self.registry_path()
        path.write_text(json.dumps(registry.to_dict(), indent=2, sort_keys=True))


__all__ = [
    "ArtifactRegistry",
    "StrategyArtifactError",
    "StrategyArtifactStore",
]
