"""Data models used by the configuration guardian."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bot_v2.config.types import Profile
from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.config import ConfigBaselinePayload

from .logging_utils import logger


@dataclass
class DriftEvent:
    """Record of a configuration drift incident."""

    timestamp: datetime
    component: str
    drift_type: str
    severity: str
    details: dict[str, Any]
    suggested_response: str
    applied_response: str
    resolution_notes: str | None = None


@dataclass
class BaselineSnapshot:
    """Immutable snapshot of configuration and trading state."""

    timestamp: datetime
    config_dict: dict[str, Any]
    config_hash: str
    env_keys: set[str]
    critical_env_values: dict[str, str]
    active_symbols: list[str]
    open_positions: dict[str, dict[str, Any]]
    account_equity: Decimal | None
    total_exposure: Decimal
    profile: Profile
    broker_type: str
    risk_limits: dict[str, Any]

    def validate_config_against_state(
        self,
        new_config_dict: dict[str, Any],
        current_balances: list[Balance],
        current_positions: list[Position],
        current_equity: Decimal | None,
    ) -> list[DriftEvent]:
        """Validate proposed config changes against live trading state."""
        from .state_validator import StateValidator

        return StateValidator(self).validate_config_against_state(
            new_config_dict, current_balances, current_positions, current_equity
        )

    @staticmethod
    def create(
        config_dict: dict[str, Any] | ConfigBaselinePayload,
        active_symbols: list[str],
        positions: list[Position],
        account_equity: Decimal | None,
        profile: Profile,
        broker_type: str,
        *,
        settings: RuntimeSettings | None = None,
    ) -> BaselineSnapshot:
        """Factory method to create baseline snapshot at startup."""
        if isinstance(config_dict, ConfigBaselinePayload):
            payload_dict = config_dict.to_dict()
        else:
            payload_dict = dict(config_dict)

        resolved_active_symbols = (
            active_symbols.copy() if active_symbols else list(payload_dict.get("symbols", []))
        )

        total_exposure = Decimal("0")
        position_summaries: dict[str, dict[str, float]] = {}

        for pos in positions:
            if hasattr(pos, "symbol") and hasattr(pos, "size") and hasattr(pos, "price"):
                symbol = pos.symbol
                size = abs(float(pos.size))
                price = float(pos.price)
                exposure = Decimal(str(size * price))
                total_exposure += exposure

                position_summaries[symbol] = {
                    "size": size,
                    "price": price,
                    "exposure": float(exposure),
                }

        normalized_config = {
            key: value for key, value in payload_dict.items() if key not in {"metadata"}
        }
        config_hash_payload = json.dumps(
            normalized_config,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        config_hash = hashlib.sha256(config_hash_payload).hexdigest()

        env_keys = set()
        critical_env_values = {}
        critical_env_vars = {
            "COINBASE_DEFAULT_QUOTE",
            "PERPS_ENABLE_STREAMING",
            "ORDER_PREVIEW_ENABLED",
        }

        resolved_settings = settings or load_runtime_settings()

        for var in critical_env_vars:
            value = resolved_settings.raw_env.get(var)
            if value is not None:
                env_keys.add(var)
                if not var.upper().endswith(("_KEY", "_SECRET", "_TOKEN")):
                    critical_env_values[var] = value

        risk_limits = {
            "max_position_size": payload_dict.get("max_position_size", "1000"),
            "max_leverage": payload_dict.get("max_leverage", 3),
            "daily_loss_limit": payload_dict.get("daily_loss_limit", "0"),
        }

        logger.info(
            "Baseline snapshot created",
            operation="config_guardian",
            stage="create_baseline",
            active_symbols=len(resolved_active_symbols),
            total_exposure=float(total_exposure),
        )

        return BaselineSnapshot(
            timestamp=datetime.now(UTC),
            config_dict=payload_dict.copy(),
            config_hash=str(config_hash),
            env_keys=env_keys,
            critical_env_values=critical_env_values,
            active_symbols=resolved_active_symbols,
            open_positions=position_summaries,
            account_equity=account_equity,
            total_exposure=total_exposure,
            profile=profile,
            broker_type=broker_type,
            risk_limits=risk_limits,
        )


__all__ = ["DriftEvent", "BaselineSnapshot"]
