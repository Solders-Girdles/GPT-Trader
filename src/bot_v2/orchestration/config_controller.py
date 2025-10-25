"""Configuration lifecycle helpers for the trading bot orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.orchestration.configuration import BotConfig, ConfigManager
from bot_v2.orchestration.runtime_settings import RuntimeSettings
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource
from bot_v2.utilities.config import ConfigBaselinePayload


@dataclass
class ConfigChange:
    """Tracks detected configuration updates and the computed diff."""

    updated: BotConfig
    diff: dict[str, Any]


class ConfigController:
    """Owns the active bot configuration and synchronizes runtime state."""

    def __init__(
        self,
        config: BotConfig,
        *,
        settings: RuntimeSettings | None = None,
        reduce_only_state_manager: Any = None,
    ) -> None:
        self._manager = ConfigManager.from_config(config, settings=settings)
        self._pending_change: ConfigChange | None = None
        self._reduce_only_state_manager = reduce_only_state_manager
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

    def _set_current_config(self, config: BotConfig) -> None:
        self._manager.replace_config(config)
        self._reduce_only_mode_state = bool(config.reduce_only_mode)

        # Update state manager if available
        if self._reduce_only_state_manager is not None:
            self._reduce_only_state_manager.set_reduce_only_mode(
                enabled=config.reduce_only_mode,
                reason="config_update",
                source=ReduceOnlyModeSource.CONFIG,
                metadata={"config_source": "config_update"},
            )

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

        reduce_only_current = bool(self.current.reduce_only_mode)
        reduce_only_risk = bool(risk_manager.is_reduce_only_mode())
        reduce_only = reduce_only_current or reduce_only_risk
        if reduce_only != self.current.reduce_only_mode:
            updated = self.current.with_overrides(reduce_only_mode=reduce_only)
            self._set_current_config(updated)
        else:
            self._reduce_only_mode_state = reduce_only

    def set_reduce_only_mode(
        self, enabled: bool, *, reason: str, risk_manager: LiveRiskManager | None = None
    ) -> bool:
        """Toggle reduce-only mode, updating both config and risk manager."""

        if enabled == self._reduce_only_mode_state:
            return False

        # Use state manager if available
        if self._reduce_only_state_manager is not None:
            changed = self._reduce_only_state_manager.set_reduce_only_mode(
                enabled=enabled,
                reason=reason,
                source=ReduceOnlyModeSource.CONFIG,
                metadata={"requested_by": "config_controller"},
            )
            if changed:
                updated = self.current.with_overrides(reduce_only_mode=enabled)
                self._manager.replace_config(updated)
                self._reduce_only_mode_state = enabled
                if risk_manager is not None:
                    risk_manager.set_reduce_only_mode(enabled, reason=reason)
            return changed
        else:
            # Fallback to legacy behavior
            updated = self.current.with_overrides(reduce_only_mode=enabled)
            self._set_current_config(updated)
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

        # Use state manager if available
        if self._reduce_only_state_manager is not None:
            changed = self._reduce_only_state_manager.set_reduce_only_mode(
                enabled=enabled,
                reason="risk_update",
                source=ReduceOnlyModeSource.RISK_MANAGER,
                metadata={"requested_by": "config_controller"},
            )
            if changed:
                updated = self.current.with_overrides(reduce_only_mode=enabled)
                self._manager.replace_config(updated)
                self._reduce_only_mode_state = enabled
            return changed
        else:
            # Fallback to legacy behavior
            updated = self.current.with_overrides(reduce_only_mode=enabled)
            self._set_current_config(updated)
            return True

    # ------------------------------------------------------------------
    def _summarize_diff(self, current: BotConfig, updated: BotConfig) -> dict[str, Any]:
        current_payload = ConfigBaselinePayload.from_config(
            current,
            derivatives_enabled=bool(getattr(current, "derivatives_enabled", False)),
        )
        updated_payload = ConfigBaselinePayload.from_config(
            updated,
            derivatives_enabled=bool(getattr(updated, "derivatives_enabled", False)),
        )

        return current_payload.diff(updated_payload)
