"""Profile-specific configuration builders."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import time
from decimal import Decimal

from bot_v2.config import path_registry
from bot_v2.config.types import Profile

from .core import BotConfig

ConfigFactory = Callable[..., BotConfig]

logger = logging.getLogger(__name__)


def build_profile_config(profile: Profile, create_config: ConfigFactory) -> BotConfig:
    """Construct a configuration tailored to the requested profile."""

    if profile == Profile.CANARY:
        return build_canary_config(profile, create_config)
    if profile == Profile.DEV:
        return create_config(
            profile=profile,
            mock_broker=True,
            mock_fills=True,
            max_position_size=Decimal("10000"),
            dry_run=True,
        )
    if profile == Profile.DEMO:
        return create_config(
            profile=profile,
            max_position_size=Decimal("100"),
            max_leverage=1,
            enable_shorts=False,
        )
    if profile == Profile.SPOT:
        return create_config(
            profile=profile,
            max_position_size=Decimal("50000"),
            max_leverage=1,
            enable_shorts=False,
            mock_broker=False,
            mock_fills=False,
        )
    return build_production_config(profile, create_config)


def build_canary_config(profile: Profile, create_config: ConfigFactory) -> BotConfig:
    """Load canary profile configuration, falling back to defaults when YAML is missing."""

    symbols: Sequence[str] = ["BTC-USD"]
    max_leverage = 1
    reduce_only = True
    update_interval = 5
    trading_window_start: time | None = None
    trading_window_end: time | None = None
    trading_days: list[str] | None = None
    daily_loss_limit = Decimal("10")
    time_in_force = "IOC"

    profile_path = path_registry.PROJECT_ROOT / "config" / "profiles" / "canary.yaml"
    if profile_path.exists():
        try:
            import yaml  # type: ignore

            with profile_path.open("r") as handle:
                payload = yaml.safe_load(handle) or {}
            symbols = payload.get("trading", {}).get("symbols", symbols)
            reduce_only = payload.get("trading", {}).get("mode") == "reduce_only" or payload.get(
                "features", {}
            ).get("reduce_only_mode", True)
            max_leverage = int(payload.get("risk_management", {}).get("max_leverage", max_leverage))
            update_interval = int(
                payload.get("monitoring", {})
                .get("metrics", {})
                .get("interval_seconds", update_interval)
            )
            session = payload.get("session", {})
            start_str = session.get("start_time")
            end_str = session.get("end_time")
            trading_window_start = time.fromisoformat(start_str) if start_str else None
            trading_window_end = time.fromisoformat(end_str) if end_str else None
            trading_days = session.get(
                "days", ["monday", "tuesday", "wednesday", "thursday", "friday"]
            )
            daily_loss_limit = Decimal(
                str(payload.get("risk_management", {}).get("daily_loss_limit", daily_loss_limit))
            )
            time_in_force = payload.get("order_policy", {}).get("time_in_force", time_in_force)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to load canary profile YAML: %s", exc, exc_info=True)

    return create_config(
        profile=profile,
        symbols=list(symbols),
        reduce_only_mode=reduce_only,
        max_leverage=max_leverage,
        update_interval=update_interval,
        dry_run=False,
        max_position_size=Decimal("500"),
        trading_window_start=trading_window_start,
        trading_window_end=trading_window_end,
        trading_days=trading_days,
        daily_loss_limit=daily_loss_limit,
        time_in_force=time_in_force,
    )


def build_production_config(profile: Profile, create_config: ConfigFactory) -> BotConfig:
    """Default production configuration (perps capable)."""

    return create_config(
        profile=profile,
        max_position_size=Decimal("50000"),
        max_leverage=3,
        enable_shorts=True,
    )


__all__ = [
    "build_canary_config",
    "build_profile_config",
    "build_production_config",
]
