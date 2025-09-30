"""Configuration lifecycle helpers for the trading bot orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.orchestration.configuration import BotConfig, ConfigManager


@dataclass
class ConfigChange:
    """Tracks detected configuration updates and the computed diff."""

    updated: BotConfig
    diff: dict[str, Any]


class ConfigController:
    """Owns the active bot configuration and synchronizes runtime state."""

    _TRACKED_FIELDS = (
        "symbols",
        "update_interval",
        "max_position_size",
        "max_leverage",
        "reduce_only_mode",
        "time_in_force",
        "daily_loss_limit",
        "derivatives_enabled",
        "perps_enable_streaming",
        "perps_stream_level",
        "perps_paper_trading",
        "perps_force_mock",
        "perps_position_fraction",
        "perps_skip_startup_reconcile",
    )

    def __init__(self, config: BotConfig) -> None:
        self._manager = ConfigManager.from_config(config)
        self._pending_change: ConfigChange | None = None
        self._reduce_only_mode_state = bool(config.reduce_only_mode)

    # ------------------------------------------------------------------
    @property
    def current(self) -> BotConfig:
        config = self._manager.get_config()
        if config is None:
            raise RuntimeError("Config manager not initialized")
        return config

    @property
    def reduce_only_mode(self) -> bool:
        return bool(self._reduce_only_mode_state)

    # ------------------------------------------------------------------
    def refresh_if_changed(self) -> ConfigChange | None:
        """Rebuild configuration when inputs change and compute diffs."""

        previous = self.current
        updated = self._manager.refresh_if_changed()
        if not updated:
            return None
        diff = self._summarize_diff(previous, updated)
        change = ConfigChange(updated=updated, diff=diff)
        self._pending_change = change
        self._reduce_only_mode_state = bool(updated.reduce_only_mode)
        return change

    def consume_pending_change(self) -> ConfigChange | None:
        change = self._pending_change
        self._pending_change = None
        return change

    # ------------------------------------------------------------------
    def sync_with_risk_manager(self, risk_manager: LiveRiskManager) -> None:
        """Ensure reduce-only mode matches risk manager state."""

        reduce_only = bool(self.current.reduce_only_mode) or risk_manager.is_reduce_only_mode()
        self._reduce_only_mode_state = reduce_only
        self.current.reduce_only_mode = reduce_only

    def set_reduce_only_mode(
        self, enabled: bool, *, reason: str, risk_manager: LiveRiskManager | None = None
    ) -> bool:
        """Toggle reduce-only mode, updating both config and risk manager."""

        if enabled == self._reduce_only_mode_state:
            return False
        self._reduce_only_mode_state = enabled
        self.current.reduce_only_mode = enabled
        if risk_manager is not None:
            risk_manager.set_reduce_only_mode(enabled, reason=reason)
        return True

    def is_reduce_only_mode(self, risk_manager: LiveRiskManager | None = None) -> bool:
        if risk_manager is not None:
            if risk_manager.is_reduce_only_mode():
                return True
        return bool(self._reduce_only_mode_state)

    def apply_risk_update(self, enabled: bool) -> bool:
        if enabled == self._reduce_only_mode_state:
            return False
        self._reduce_only_mode_state = enabled
        self.current.reduce_only_mode = enabled
        return True

    # ------------------------------------------------------------------
    def _summarize_diff(self, current: BotConfig, updated: BotConfig) -> dict[str, Any]:
        diff: dict[str, Any] = {}
        for field_name in self._TRACKED_FIELDS:
            if getattr(current, field_name) != getattr(updated, field_name):
                diff[field_name] = {
                    "current": getattr(current, field_name),
                    "new": getattr(updated, field_name),
                }
        return diff
