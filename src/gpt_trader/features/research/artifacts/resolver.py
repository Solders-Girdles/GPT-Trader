"""Resolve and apply strategy artifacts to runtime configuration."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from decimal import Decimal
from typing import Any, get_args, get_origin

from gpt_trader.app.config import BotConfig
from gpt_trader.utilities.logging_patterns import get_logger

from .models import StrategyArtifact
from .store import StrategyArtifactError, StrategyArtifactStore

logger = get_logger(__name__, component="strategy_artifact_resolver")


class StrategyArtifactResolutionError(ValueError):
    """Raised when artifact resolution or approval checks fail."""


def resolve_strategy_artifact(config: BotConfig) -> StrategyArtifact | None:
    """Resolve artifact by explicit id/path (or registry when enabled)."""
    artifact_id = config.strategy_artifact_id
    artifact_path = config.strategy_artifact_path
    use_registry = config.strategy_artifact_use_registry

    if not artifact_id and not artifact_path and not use_registry:
        return None

    store = StrategyArtifactStore()

    if not artifact_id and not artifact_path and use_registry:
        profile = _profile_name(config.profile)
        if profile:
            artifact_id = store.resolve_active(profile)

    if not artifact_id and not artifact_path:
        return None

    try:
        if artifact_path:
            return store.load(artifact_path)
        return store.load(artifact_id)
    except StrategyArtifactError as exc:
        raise StrategyArtifactResolutionError(str(exc)) from exc


def apply_strategy_artifact_to_config(config: BotConfig) -> StrategyArtifact | None:
    """Apply artifact overrides to the provided BotConfig in-place."""
    if config.metadata.get("strategy_artifact_applied"):
        return None

    artifact = resolve_strategy_artifact(config)
    if artifact is None:
        return None

    if not _artifact_allowed(config, artifact):
        raise StrategyArtifactResolutionError(
            "Strategy artifact is not approved for live trading. "
            "Publish the artifact or set strategy_artifact_allow_unapproved when safe."
        )

    _apply_artifact(config, artifact)
    config.metadata["strategy_artifact_applied"] = True
    config.metadata["strategy_artifact_id"] = artifact.artifact_id
    logger.info(
        "Applied strategy artifact",
        artifact_id=artifact.artifact_id,
        strategy_type=artifact.strategy_type,
    )
    return artifact


def _artifact_allowed(config: BotConfig, artifact: StrategyArtifact) -> bool:
    if artifact.approved:
        return True
    if config.strategy_artifact_allow_unapproved:
        return True
    # Allow unapproved artifacts in non-live modes.
    if config.dry_run or config.mock_broker or config.paper_fills:
        return True
    return False


def _apply_artifact(config: BotConfig, artifact: StrategyArtifact) -> None:
    if artifact.symbols:
        config.symbols = list(artifact.symbols)
    if artifact.interval is not None:
        config.interval = int(artifact.interval)
    if artifact.strategy_type:
        config.strategy_type = artifact.strategy_type

    if artifact.strategy_parameters:
        config.strategy = _apply_dataclass_updates(config.strategy, artifact.strategy_parameters)

    if artifact.mean_reversion_parameters:
        config.mean_reversion = _apply_dataclass_updates(
            config.mean_reversion, artifact.mean_reversion_parameters
        )

    if artifact.risk_parameters:
        config.risk = _apply_dataclass_updates(config.risk, artifact.risk_parameters)

    if artifact.ensemble_parameters:
        from gpt_trader.features.live_trade.strategies.ensemble import EnsembleStrategyConfig

        base = (
            config.ensemble_config
            if isinstance(config.ensemble_config, EnsembleStrategyConfig)
            else EnsembleStrategyConfig()
        )
        config.ensemble_config = _apply_dataclass_updates(base, artifact.ensemble_parameters)

    if artifact.regime_parameters:
        try:
            from gpt_trader.features.intelligence.regime import RegimeConfig

            if isinstance(artifact.regime_parameters, RegimeConfig):
                config.regime_config = artifact.regime_parameters
            else:
                config.regime_config = RegimeConfig.from_dict(artifact.regime_parameters)
        except Exception:
            # Fall back to raw dict if parsing fails.
            config.regime_config = artifact.regime_parameters


def _apply_dataclass_updates(instance: Any, updates: dict[str, Any]) -> Any:
    if not updates:
        return instance
    if not is_dataclass(instance):
        return instance
    field_map = {field.name: field for field in fields(instance)}
    values: dict[str, Any] = {name: getattr(instance, name) for name in field_map}
    for key, value in updates.items():
        if key not in field_map:
            continue
        values[key] = _coerce_field_value(value, field_map[key].type)
    return type(instance)(**values)


def _coerce_field_value(value: Any, field_type: Any) -> Any:
    if value is None:
        return None
    if field_type in (Decimal, "Decimal"):
        return Decimal(str(value))
    origin = get_origin(field_type)
    if origin is not None:
        args = get_args(field_type)
        if Decimal in args or "Decimal" in args:
            return Decimal(str(value))
    return value


def _profile_name(profile: object | None) -> str | None:
    if profile is None:
        return None
    return getattr(profile, "value", str(profile))


__all__ = [
    "StrategyArtifactResolutionError",
    "apply_strategy_artifact_to_config",
    "resolve_strategy_artifact",
]
