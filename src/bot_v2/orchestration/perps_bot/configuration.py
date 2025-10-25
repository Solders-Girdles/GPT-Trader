"""Configuration and health management helpers."""

from __future__ import annotations

from typing import Any

from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.utilities.config import ConfigBaselinePayload

from .logging import logger


class PerpsBotConfigurationMixin:
    """Configuration lifecycle and monitoring helpers."""

    @staticmethod
    def build_baseline_snapshot(config: BotConfig, derivatives_enabled: bool) -> Any:
        """Return a baseline snapshot for configuration drift detection."""

        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        payload = ConfigBaselinePayload.from_config(
            config,
            derivatives_enabled=derivatives_enabled,
        )

        payload_dict = payload.to_dict()
        active_symbols = list(payload_dict.get("symbols") or [])
        broker_type = "mock" if config.mock_broker else "live"
        runtime_settings = getattr(getattr(config, "state", None), "runtime_settings", None)

        return _ConfigurationGuardian.create_baseline_snapshot(
            config_dict=payload_dict,
            active_symbols=active_symbols,
            positions=[],
            account_equity=None,
            profile=config.profile,
            broker_type=broker_type,
            settings=runtime_settings,
        )

    def apply_config_change(self, change: ConfigChange) -> None:
        logger.info(
            "Applying configuration change",
            operation="config_change",
            stage="apply",
            diff=change.diff,
            changed_fields=sorted(change.diff.keys()),
        )
        self.config = change.updated
        self.symbols = list(self.config.symbols or [])
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))
        self.registry = self.registry.with_updates(config=self.config)
        self.execution_coordinator.reset_order_reconciler()
        self.config_controller.sync_with_risk_manager(self.risk_manager)
        self._session_guard = TradingSessionGuard(
            start=self.config.trading_window_start,
            end=self.config.trading_window_end,
            trading_days=self.config.trading_days,
        )
        mark_windows = self._state.mark_windows
        for symbol in self.symbols:
            mark_windows.setdefault(symbol, [])
        for symbol in list(mark_windows.keys()):
            if symbol not in self.symbols:
                del mark_windows[symbol]
        self.telemetry_coordinator.init_market_services()
        self.strategy_orchestrator.init_strategy()
        self._restart_streaming_if_needed(change.diff)

        # Refresh configuration baseline to reflect the new runtime config
        new_baseline = self.build_baseline_snapshot(self.config, self._derivatives_enabled)
        self.baseline_snapshot = new_baseline
        if self.configuration_guardian is not None:
            self.configuration_guardian.reset_baseline(new_baseline)

    def _start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
        self.telemetry_coordinator.start_streaming_background()

    def _stop_streaming_background(self) -> None:
        self.telemetry_coordinator.stop_streaming_background()

    def _restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        self.telemetry_coordinator.restart_streaming_if_needed(diff)

    def _run_stream_loop(self, symbols: list[str], level: int) -> None:
        self.telemetry_coordinator._run_stream_loop(symbols, level, stop_signal=None)

    async def _run_account_telemetry(self, interval_seconds: int = 300) -> None:
        await self.telemetry_coordinator.run_account_telemetry(interval_seconds)

    def is_reduce_only_mode(self) -> bool:
        return bool(self.runtime_coordinator.is_reduce_only_mode())

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        self.runtime_coordinator.set_reduce_only_mode(enabled, reason)

    def write_health_status(self, ok: bool, message: str = "", error: str = "") -> None:
        self.system_monitor.write_health_status(ok=ok, message=message, error=error)
