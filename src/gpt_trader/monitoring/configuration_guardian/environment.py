"""Environment monitor for runtime configuration guardian."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from gpt_trader.app.config import BotConfig

from .base import ConfigurationMonitor
from .models import DriftEvent
from .responses import DriftResponse


class EnvironmentMonitor(ConfigurationMonitor):
    """Monitors critical environment variables for runtime changes."""

    CRITICAL_ENV_VARS = {
        "COINBASE_ENABLE_DERIVATIVES",
        "SPOT_FORCE_LIVE",
        "PERPS_ENABLE_STREAMING",
    }

    RISK_ENV_VARS = {
        "PERPS_POSITION_FRACTION",
        "ORDER_PREVIEW_ENABLED",
    }

    MONITOR_ENV_VARS = {
        "COINBASE_DEFAULT_QUOTE",
        "MOCK_BROKER",
        "PERPS_PAPER",
    }

    def __init__(
        self,
        baseline_snapshot: Any,
        *,
        config: BotConfig | None = None,
    ) -> None:
        self.baseline = baseline_snapshot
        self._config = config
        self._last_state = self._capture_current_state()

    def update_baseline(self, baseline_snapshot: Any) -> None:
        """Refresh cached state after a baseline change."""
        self.baseline = baseline_snapshot
        self._last_state = self._capture_current_state()

    def check_changes(self) -> list[DriftEvent]:
        """Check for environment variable changes."""
        events: list[DriftEvent] = []
        current_state = self._capture_current_state()

        for var in self.CRITICAL_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(
                    DriftEvent(
                        timestamp=datetime.now(UTC),
                        component=self.monitor_name,
                        drift_type="critical_env_changed",
                        severity="critical",
                        details={
                            "variable": var,
                            "old_value": old_val,
                            "new_value": new_val,
                            "impact": "Runtime change prevents safe operation",
                        },
                        suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                        applied_response=DriftResponse.EMERGENCY_SHUTDOWN,
                    )
                )

        for var in self.RISK_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(
                    DriftEvent(
                        timestamp=datetime.now(UTC),
                        component=self.monitor_name,
                        drift_type="risk_env_changed",
                        severity="high",
                        details={
                            "variable": var,
                            "old_value": old_val,
                            "new_value": new_val,
                            "impact": "Changes risk management behavior",
                        },
                        suggested_response=DriftResponse.REDUCE_ONLY,
                        applied_response=DriftResponse.REDUCE_ONLY,
                    )
                )

        for var in self.MONITOR_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(
                    DriftEvent(
                        timestamp=datetime.now(UTC),
                        component=self.monitor_name,
                        drift_type="monitored_env_changed",
                        severity="low",
                        details={
                            "variable": var,
                            "old_value": old_val,
                            "new_value": new_val,
                            "impact": "Informational only",
                        },
                        suggested_response=DriftResponse.STICKY,
                        applied_response=DriftResponse.STICKY,
                    )
                )

        self._last_state = current_state
        return events

    def get_current_state(self) -> dict[str, Any]:
        """Get current environment monitor state."""
        return self._capture_current_state()

    @property
    def monitor_name(self) -> str:
        return "environment_monitor"

    def _capture_current_state(self) -> dict[str, Any]:
        """Capture current environment state (safe values only)."""
        state: dict[str, Any] = {}
        all_vars = self.CRITICAL_ENV_VARS | self.RISK_ENV_VARS | self.MONITOR_ENV_VARS

        for var in all_vars:
            value = os.environ.get(var)
            if value is not None:
                if var.upper().endswith(("_KEY", "_SECRET", "_TOKEN")):
                    state[var] = "[REDACTED]"
                else:
                    state[var] = value

        return state


__all__ = ["EnvironmentMonitor"]
