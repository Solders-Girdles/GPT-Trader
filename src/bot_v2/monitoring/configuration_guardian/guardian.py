"""Main configuration guardian orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bot_v2.config.schemas import ConfigValidationResult
from bot_v2.config.types import Profile
from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.config import ConfigBaselinePayload

from .detector import DriftDetector
from .environment import EnvironmentMonitor
from .logging_utils import logger
from .models import BaselineSnapshot, DriftEvent
from .responses import DriftResponse
from .state_validator import StateValidator


class ConfigurationGuardian:
    """Main guardian class that orchestrates runtime configuration validation."""

    def __init__(
        self,
        baseline_snapshot: BaselineSnapshot,
        *,
        settings: RuntimeSettings | None = None,
        event_store: EventStore | None = None,
        bot_id: str = "config_guardian",
    ) -> None:
        self.baseline = baseline_snapshot
        self._settings = settings or load_runtime_settings()
        self._event_store = event_store or EventStore()
        self._bot_id = bot_id

        self.environment_monitor = EnvironmentMonitor(baseline_snapshot, settings=self._settings)
        self.state_validator = StateValidator(baseline_snapshot)
        self.drift_detector = DriftDetector(baseline_snapshot)

        self.monitors = [self.environment_monitor, self.drift_detector]

        logger.info(
            "ConfigurationGuardian initialized",
            operation="config_guardian",
            stage="init",
            baseline_timestamp=baseline_snapshot.timestamp.isoformat(),
        )

    def reset_baseline(self, new_snapshot: BaselineSnapshot, *, user_id: str | None = None) -> None:
        """Reset baseline snapshot following an intentional configuration change."""
        old_config = self.baseline.config_dict
        new_config = new_snapshot.config_dict

        changes: dict[str, tuple[Any, Any]] = {}
        for key in set(old_config.keys()) | set(new_config.keys()):
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                changes[key] = (old_val, new_val)

        if changes:
            self._log_config_delta("baseline_reset", changes, user_id=user_id)

        self.baseline = new_snapshot
        self.environment_monitor.update_baseline(new_snapshot)
        self.state_validator.update_baseline(new_snapshot)
        self.drift_detector.update_baseline(new_snapshot)
        logger.info(
            "ConfigurationGuardian baseline reset",
            operation="config_guardian",
            stage="reset_baseline",
            baseline_timestamp=new_snapshot.timestamp.isoformat(),
            user_id=user_id,
            changes_count=len(changes),
        )

    def pre_cycle_check(
        self,
        proposed_config_dict: dict[str, Any] | None = None,
        current_balances: list[Balance] | None = None,
        current_positions: list[Position] | None = None,
        current_equity: Decimal | None = None,
    ) -> ConfigValidationResult:
        """Master validation method called before each trading cycle."""
        all_events: list[DriftEvent] = []

        for monitor in self.monitors:
            try:
                events = monitor.check_changes()
                all_events.extend(events)
            except Exception as exc:
                logger.error(
                    "Monitor check failed",
                    operation="config_guardian",
                    stage="monitor_check",
                    monitor=monitor.monitor_name,
                    error=str(exc),
                    exc_info=True,
                )
                all_events.append(
                    DriftEvent(
                        timestamp=datetime.now(UTC),
                        component=monitor.monitor_name,
                        drift_type="monitor_failure",
                        severity="high",
                        details={"error": str(exc)},
                        suggested_response=DriftResponse.REDUCE_ONLY,
                        applied_response=DriftResponse.REDUCE_ONLY,
                    )
                )

        if proposed_config_dict:
            state_events = self.state_validator.validate_config_against_state(
                proposed_config_dict,
                current_balances or [],
                current_positions or [],
                current_equity,
            )
            all_events.extend(state_events)

        self.drift_detector.record_drift_events(all_events)

        critical_events = [e for e in all_events if e.severity == "critical"]
        high_events = [e for e in all_events if e.severity == "high"]

        if critical_events or high_events:
            error_messages = []
            for event in critical_events:
                error_messages.append(
                    f"[{event.component}] {event.drift_type} (severity=critical, response=emergency_shutdown): {event.details.get('message', '')}"
                )
            for event in high_events:
                error_messages.append(
                    f"[{event.component}] {event.drift_type} (severity=high): {event.details.get('message', '')}"
                )

            return ConfigValidationResult(is_valid=False, errors=error_messages, warnings=[])

        warning_messages = [
            f"[{e.component}] {e.drift_type}: {e.details.get('message', '')}" for e in all_events
        ]

        return ConfigValidationResult(is_valid=True, errors=[], warnings=warning_messages)

    def get_health_status(self) -> dict[str, Any]:
        """Get guardian health status for monitoring."""
        monitors_status: dict[str, str] = {}
        status: dict[str, Any] = {
            "baseline_timestamp": self.baseline.timestamp.isoformat(),
            "monitors_status": monitors_status,
            "drift_summary": self.drift_detector.get_drift_summary(),
        }

        for monitor in self.monitors:
            try:
                monitors_status[monitor.monitor_name] = "healthy"
            except Exception as exc:
                monitors_status[monitor.monitor_name] = f"error: {exc}"
                logger.error(
                    "Monitor health check failed",
                    operation="config_guardian",
                    stage="health_check",
                    monitor=monitor.monitor_name,
                    error=str(exc),
                    exc_info=True,
                )

        return status

    def _log_config_delta(
        self,
        change_type: str,
        changes: dict[str, tuple[Any, Any]],
        *,
        user_id: str | None = None,
    ) -> None:
        """Log configuration delta to event store."""
        event_data = {
            "event_type": "config_delta",
            "change_type": change_type,
            "changes": {
                field: {"old": str(old), "new": str(new)} for field, (old, new) in changes.items()
            },
            "user_id": user_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._event_store.append_metric(self._bot_id, event_data)

        logger.info(
            "Logged config delta",
            operation="config_guardian",
            stage="log_delta",
            change_type=change_type,
            field_count=len(changes),
        )

    @staticmethod
    def create_baseline_snapshot(
        config_dict: dict[str, Any] | ConfigBaselinePayload,
        active_symbols: list[str],
        positions: list[Position],
        account_equity: Decimal | None,
        profile: Profile,
        broker_type: str,
        *,
        settings: RuntimeSettings | None = None,
    ) -> BaselineSnapshot:
        """Proxy to BaselineSnapshot factory for backward compatibility."""
        return BaselineSnapshot.create(
            config_dict,
            active_symbols,
            positions,
            account_equity,
            profile,
            broker_type,
            settings=settings,
        )


__all__ = ["ConfigurationGuardian"]
