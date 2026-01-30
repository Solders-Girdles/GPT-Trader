"""Strategy artifact contract for research -> live handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class StrategyArtifact:
    """Immutable snapshot of strategy configuration and validation evidence."""

    artifact_id: str
    strategy_type: str
    created_at: str
    source: str | None = None
    symbols: list[str] = field(default_factory=list)
    interval: int | None = None
    strategy_parameters: dict[str, Any] = field(default_factory=dict)
    mean_reversion_parameters: dict[str, Any] = field(default_factory=dict)
    ensemble_parameters: dict[str, Any] = field(default_factory=dict)
    regime_parameters: dict[str, Any] = field(default_factory=dict)
    risk_parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)
    approved: bool = False
    approved_at: str | None = None
    approved_by: str | None = None
    notes: str | None = None
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        strategy_type: str,
        symbols: list[str],
        interval: int | None,
        strategy_parameters: dict[str, Any] | None = None,
        mean_reversion_parameters: dict[str, Any] | None = None,
        ensemble_parameters: dict[str, Any] | None = None,
        regime_parameters: dict[str, Any] | None = None,
        risk_parameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        evidence_paths: list[str] | None = None,
        source: str | None = None,
        notes: str | None = None,
    ) -> StrategyArtifact:
        """Create a new strategy artifact with a generated id and timestamp."""
        created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            artifact_id=f"artifact-{uuid4().hex[:12]}",
            strategy_type=strategy_type,
            created_at=created_at,
            source=source,
            symbols=list(symbols),
            interval=interval,
            strategy_parameters=strategy_parameters or {},
            mean_reversion_parameters=mean_reversion_parameters or {},
            ensemble_parameters=ensemble_parameters or {},
            regime_parameters=regime_parameters or {},
            risk_parameters=risk_parameters or {},
            metrics=metrics or {},
            validation=validation or {},
            evidence_paths=list(evidence_paths or []),
            approved=False,
            notes=notes,
            schema_version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "strategy_type": self.strategy_type,
            "created_at": self.created_at,
            "source": self.source,
            "symbols": list(self.symbols),
            "interval": self.interval,
            "strategy_parameters": self.strategy_parameters,
            "mean_reversion_parameters": self.mean_reversion_parameters,
            "ensemble_parameters": self.ensemble_parameters,
            "regime_parameters": self.regime_parameters,
            "risk_parameters": self.risk_parameters,
            "metrics": self.metrics,
            "validation": self.validation,
            "evidence_paths": list(self.evidence_paths),
            "approved": self.approved,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyArtifact:
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            strategy_type=str(data.get("strategy_type") or ""),
            created_at=str(data.get("created_at") or ""),
            source=data.get("source"),
            symbols=list(data.get("symbols") or []),
            interval=data.get("interval"),
            strategy_parameters=dict(data.get("strategy_parameters") or {}),
            mean_reversion_parameters=dict(data.get("mean_reversion_parameters") or {}),
            ensemble_parameters=dict(data.get("ensemble_parameters") or {}),
            regime_parameters=dict(data.get("regime_parameters") or {}),
            risk_parameters=dict(data.get("risk_parameters") or {}),
            metrics=dict(data.get("metrics") or {}),
            validation=dict(data.get("validation") or {}),
            evidence_paths=list(data.get("evidence_paths") or []),
            approved=bool(data.get("approved", False)),
            approved_at=data.get("approved_at"),
            approved_by=data.get("approved_by"),
            notes=data.get("notes"),
            schema_version=int(data.get("schema_version", 1)),
        )


__all__ = ["StrategyArtifact"]
