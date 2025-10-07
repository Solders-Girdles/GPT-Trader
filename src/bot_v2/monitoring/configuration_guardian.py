"""Runtime configuration drift detection and trading state validation."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Set

from bot_v2.config.schemas import ConfigValidationResult
from bot_v2.config.types import Profile
from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.monitoring.interfaces import TradeMetrics

logger = logging.getLogger(__name__)


class DriftResponse:
    """Response strategies for configuration drift detection."""

    STICKY = "sticky"  # Keep old config, warn, continue
    REDUCE_ONLY = "reduce_only"  # Switch to reduce-only mode
    EMERGENCY_SHUTDOWN = "emergency_shutdown"  # Stop all trading immediately


@dataclass
class DriftEvent:
    """Record of a configuration drift incident."""

    timestamp: datetime
    component: str  # "environment" | "file" | "state_validator"
    drift_type: str  # e.g., "env_var_changed", "config_file_modified", "unsafe_leverage"
    severity: str  # "low" | "medium" | "high" | "critical"
    details: Dict[str, Any]
    suggested_response: str  # DriftResponse value
    applied_response: str
    resolution_notes: Optional[str] = None


@dataclass
class BaselineSnapshot:
    """Immutable snapshot of configuration and trading state."""

    timestamp: datetime

    def validate_config_against_state(
        self,
        new_config_dict: Dict[str, Any],
        current_balances: List[Balance],
        current_positions: List[Position],
        current_equity: Optional[Decimal]
    ) -> List[DriftEvent]:
        """Validate proposed config changes against live trading state."""
        return StateValidator(self).validate_config_against_state(
            new_config_dict, current_balances, current_positions, current_equity
        )

    # Configuration
    config_dict: Dict[str, Any]
    config_hash: str  # For quick comparison

    # Environment variables (secure version - no secrets)
    env_keys: Set[str]  # Which env vars were set
    critical_env_values: Dict[str, str]  # Safe env vars only (not secrets)

    # Trading state
    active_symbols: List[str]
    open_positions: Dict[str, Dict[str, Any]]  # symbol -> position_summary
    account_equity: Optional[Decimal]
    total_exposure: Decimal

    # Contextual info
    profile: Profile
    broker_type: str
    risk_limits: Dict[str, Any]


class ConfigurationMonitor(ABC):
    """Abstract base for configuration monitoring components."""

    @abstractmethod
    def check_changes(self) -> List[DriftEvent]:
        """Check for configuration changes, return drift events if found."""
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state for monitoring."""
        pass

    @property
    @abstractmethod
    def monitor_name(self) -> str:
        """Component name for logging."""
        pass


class EnvironmentMonitor(ConfigurationMonitor):
    """Monitors critical environment variables for runtime changes.

    Watches these signals:
    - COINBASE_ENABLE_DERIVATIVES (trading mode switch)
    - COINBASE_DEFAULT_QUOTE (currency changes)
    - PERPS_ENABLE_STREAMING (network behavior)
    - PERPS_POSITION_FRACTION (risk limit changes)
    - ORDER_PREVIEW_ENABLED (execution behavior)
    - SPOT_FORCE_LIVE (broker switch from mock to live)

    Ignores changing credentials/keys during runtime.
    """

    # Environment variables that should NEVER change during runtime
    CRITICAL_ENV_VARS = {
        "COINBASE_ENABLE_DERIVATIVES",  # Trading mode switch
        "SPOT_FORCE_LIVE",  # Mock vs live broker switch
        "PERPS_ENABLE_STREAMING",  # Network behavior change
    }

    # Environment variables that require reduce-only response
    RISK_ENV_VARS = {
        "PERPS_POSITION_FRACTION",  # Position sizing change
        "ORDER_PREVIEW_ENABLED",  # Execution behavior change
    }

    # Environment variables that require attention but allow sticky
    MONITOR_ENV_VARS = {
        "COINBASE_DEFAULT_QUOTE",  # Currency changes
        "PERPS_FORCE_MOCK",  # Fallback broker changes
        "PERPS_PAPER",  # Paper trading toggles
    }

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot
        self._last_state = self._capture_current_state()

    def check_changes(self) -> List[DriftEvent]:
        """Check for environment variable changes."""
        events = []
        current_state = self._capture_current_state()

        # Check critical vars that should never change
        for var in self.CRITICAL_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="critical_env_changed",
                    severity="critical",
                    details={
                        "variable": var,
                        "old_value": old_val,
                        "new_value": new_val,
                        "impact": "Runtime change prevents safe operation"
                    },
                    suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                    applied_response=DriftResponse.EMERGENCY_SHUTDOWN
                ))

        # Check risk vars that modify trading behavior
        for var in self.RISK_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="risk_env_changed",
                    severity="high",
                    details={
                        "variable": var,
                        "old_value": old_val,
                        "new_value": new_val,
                        "impact": "Changes risk management behavior"
                    },
                    suggested_response=DriftResponse.REDUCE_ONLY,
                    applied_response=DriftResponse.REDUCE_ONLY
                ))

        # Check monitored vars for informational purposes
        for var in self.MONITOR_ENV_VARS:
            old_val = self._last_state.get(var)
            new_val = current_state.get(var)
            if old_val != new_val:
                events.append(DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="monitored_env_changed",
                    severity="low",
                    details={
                        "variable": var,
                        "old_value": old_val,
                        "new_value": new_val,
                        "impact": "Informational only"
                    },
                    suggested_response=DriftResponse.STICKY,
                    applied_response=DriftResponse.STICKY
                ))

        self._last_state = current_state
        return events

    def get_current_state(self) -> Dict[str, Any]:
        """Get current environment monitor state."""
        return self._capture_current_state()

    @property
    def monitor_name(self) -> str:
        return "environment_monitor"

    def _capture_current_state(self) -> Dict[str, Any]:
        """Capture current environment state (safe values only)."""
        state = {}
        all_vars = self.CRITICAL_ENV_VARS | self.RISK_ENV_VARS | self.MONITOR_ENV_VARS

        for var in all_vars:
            value = os.getenv(var)
            if value is not None:
                # Don't store sensitive values
                if var.upper().endswith(('_KEY', '_SECRET', '_TOKEN')):
                    state[var] = "[REDACTED]"
                else:
                    state[var] = value

        return state


class StateValidator(ConfigurationMonitor):
    """Validates configuration changes against current trading state.

    Enforces these invariants:
    1. Symbol universe cannot remove active positions without closing them
    2. Leverage changes cannot cause immediate liquidation
    3. Position size limits respect current exposure
    4. Profile changes must be compatible with current state
    """

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot

    def check_changes(self) -> List[DriftEvent]:
        """Validate configuration state against trading invariants."""
        # This will be called with new config proposals during runtime
        # For now, return empty list as we need the trading state context
        return []

    def validate_config_against_state(
        self,
        new_config_dict: Dict[str, Any],
        current_balances: List[Balance],
        current_positions: List[Position],
        current_equity: Optional[Decimal]
    ) -> List[DriftEvent]:
        """Validate proposed config changes against live trading state."""

        events = []

        # Extract key values from config
        new_symbols = set(new_config_dict.get("symbols", []))
        new_max_leverage = new_config_dict.get("max_leverage", 3)
        new_position_size = new_config_dict.get("max_position_size", Decimal("1000"))
        new_profile = new_config_dict.get("profile")

        # Check: Symbol universe changes
        baseline_symbols = set(self.baseline.active_symbols)
        removed_symbols = baseline_symbols - new_symbols
        added_symbols = new_symbols - baseline_symbols

        # Don't allow removing symbols with active positions
        active_symbols = {pos.symbol for pos in current_positions if hasattr(pos, 'symbol')}
        removed_with_positions = removed_symbols & active_symbols

        if removed_with_positions:
            events.append(DriftEvent(
                timestamp=datetime.now(UTC),
                component=self.monitor_name,
                drift_type="symbols_remove_active_positions",
                severity="critical",
                details={
                    "removed_symbols": list(removed_with_positions),
                    "message": f"Cannot remove {removed_with_positions} - active positions exist"
                },
                suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                applied_response=DriftResponse.EMERGENCY_SHUTDOWN
            ))

        # Check: Leverage changes vs current positions
        current_leverage = self._calculate_current_leverage(current_positions, current_equity)
        if current_leverage > new_max_leverage:
            events.append(DriftEvent(
                timestamp=datetime.now(UTC),
                component=self.monitor_name,
                drift_type="leverage_violation_current_positions",
                severity="high",
                details={
                    "current_leverage": float(current_leverage),
                    "new_max_leverage": new_max_leverage,
                    "message": "Current positions exceed new leverage limit"
                },
                suggested_response=DriftResponse.REDUCE_ONLY,
                applied_response=DriftResponse.REDUCE_ONLY
            ))

        # Check: Position size vs current exposure
        current_exposure = sum(
            abs(float(getattr(pos, 'size', 0))) * float(getattr(pos, 'price', 0))
            for pos in current_positions
            if hasattr(pos, 'size') and hasattr(pos, 'price')
        )

        if current_exposure > float(new_position_size):
            events.append(DriftEvent(
                timestamp=datetime.now(UTC),
                component=self.monitor_name,
                drift_type="position_size_violation_current_exposure",
                severity="high",
                details={
                    "current_exposure": current_exposure,
                    "new_max_position_size": float(new_position_size),
                    "message": "Current exposure exceeds new position limit"
                },
                suggested_response=DriftResponse.REDUCE_ONLY,
                applied_response=DriftResponse.REDUCE_ONLY
            ))

        # Check: Profile compatibility
        if str(new_profile) != str(self.baseline.profile):
            # Profile changes during runtime are critical
            events.append(DriftEvent(
                timestamp=datetime.now(UTC),
                component=self.monitor_name,
                drift_type="profile_changed_during_runtime",
                severity="critical",
                details={
                    "old_profile": str(self.baseline.profile),
                    "new_profile": str(new_profile),
                    "message": "Profile changes during runtime not supported"
                },
                suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                applied_response=DriftResponse.EMERGENCY_SHUTDOWN
            ))

        return events

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state validator status."""
        return {"baseline_snapshot_timestamp": self.baseline.timestamp.isoformat()}

    @property
    def monitor_name(self) -> str:
        return "state_validator"

    def _calculate_current_leverage(
        self,
        positions: List[Position],
        equity: Optional[Decimal]
    ) -> Decimal:
        """Calculate current leverage across all positions."""
        if not equity or equity <= 0:
            return Decimal("0")

        total_exposure = Decimal("0")
        for pos in positions:
            if hasattr(pos, 'size') and hasattr(pos, 'price'):
                size = abs(float(pos.size))
                price = float(pos.price)
                exposure = Decimal(str(size * price))
                total_exposure += exposure

        return (total_exposure / equity) if total_exposure > 0 else Decimal("0")


class DriftDetector(ConfigurationMonitor):
    """Detects configuration drift from baseline snapshot.

    Baseline snapshot captures the 'safe known state' at bot startup.
    Any deviation from this baseline triggers drift detection.
    """

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot
        self.drift_history: List[DriftEvent] = []

    def check_changes(self) -> List[DriftEvent]:
        """Compare current state against baseline."""
        # This method is called by the guardian to check all monitors
        # Individual monitor check_changes() methods handle the actual detection
        return self.drift_history

    def record_drift_events(self, events: List[DriftEvent]) -> None:
        """Record drift events for audit trail."""
        self.drift_history.extend(events)
        logger.info(f"Recorded {len(events)} drift events")

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift activity."""
        if not self.drift_history:
            return {"total_events": 0, "highest_severity": "none"}

        severities = [e.severity for e in self.drift_history]
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        return {
            "total_events": len(self.drift_history),
            "highest_severity": max(severities, key=lambda s: severity_order.get(s, -1)),
            "events_by_component": {
                comp: len([e for e in self.drift_history if e.component == comp])
                for comp in set(e.component for e in self.drift_history)
            }
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current drift detector state."""
        return self.get_drift_summary()

    @property
    def monitor_name(self) -> str:
        return "drift_detector"


class ConfigurationGuardian:
    """Main guardian class that orchestrates runtime configuration validation.

    Integrates all monitoring components and provides the external API
    for checking configuration safety before trading cycles.
    """

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot

        # Initialize monitoring components
        self.environment_monitor = EnvironmentMonitor(baseline_snapshot)
        self.state_validator = StateValidator(baseline_snapshot)
        self.drift_detector = DriftDetector(baseline_snapshot)

        self.monitors = [
            self.environment_monitor,
            self.drift_detector
        ]

        logger.info(f"ConfigurationGuardian initialized with baseline from {baseline_snapshot.timestamp}")

    def pre_cycle_check(
        self,
        proposed_config_dict: Optional[Dict[str, Any]] = None,
        current_balances: Optional[List[Balance]] = None,
        current_positions: Optional[List[Position]] = None,
        current_equity: Optional[Decimal] = None
    ) -> ConfigValidationResult:
        """Master validation method called before each trading cycle."""

        all_events = []

        # Check environment and file changes
        for monitor in self.monitors:
            try:
                events = monitor.check_changes()
                all_events.extend(events)
            except Exception as e:
                logger.error(f"Monitor {monitor.monitor_name} failed: {e}")
                all_events.append(DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=monitor.monitor_name,
                    drift_type="monitor_failure",
                    severity="high",
                    details={"error": str(e)},
                    suggested_response=DriftResponse.REDUCE_ONLY,
                    applied_response=DriftResponse.REDUCE_ONLY
                ))

        # If new config is proposed, validate against current state
        if proposed_config_dict:
            state_events = self.state_validator.validate_config_against_state(
                proposed_config_dict,
                current_balances or [],
                current_positions or [],
                current_equity
            )
            all_events.extend(state_events)

        # Record all events for audit trail
        self.drift_detector.record_drift_events(all_events)

        # Determine overall validation result
        critical_events = [e for e in all_events if e.severity == "critical"]
        high_events = [e for e in all_events if e.severity == "high"]

        # Fail validation for both critical AND high-severity events
        # Critical → Emergency shutdown
        # High → Reduce-only mode
        if critical_events or high_events:
            error_messages = []
            for event in critical_events + high_events:
                error_messages.append(
                    f"[{event.component}] {event.drift_type}: {event.details.get('message', '')}"
                )

            return ConfigValidationResult(
                is_valid=False,
                errors=error_messages,
                warnings=[]
            )

        # Valid but may have warnings (low/medium severity)
        warning_messages = [f"[{e.component}] {e.drift_type}: {e.details.get('message', '')}"
                          for e in all_events]

        return ConfigValidationResult(
            is_valid=True,
            errors=[],
            warnings=warning_messages
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get guardian health status for monitoring."""
        status = {
            "baseline_timestamp": self.baseline.timestamp.isoformat(),
            "monitors_status": {},
            "drift_summary": self.drift_detector.get_drift_summary()
        }

        for monitor in self.monitors:
            try:
                status["monitors_status"][monitor.monitor_name] = "healthy"
                # Could add more detailed health checks here
            except Exception as e:
                status["monitors_status"][monitor.monitor_name] = f"error: {e}"
                logger.error(f"Monitor {monitor.monitor_name} health check failed: {e}")

        return status

    def create_baseline_snapshot(
        config_dict: Dict[str, Any],
        active_symbols: List[str],
        positions: List[Position],
        account_equity: Optional[Decimal],
        profile: Profile,
        broker_type: str
    ) -> BaselineSnapshot:
        """Factory method to create baseline snapshot at startup."""

        # Calculate total exposure
        total_exposure = Decimal("0")
        position_summaries = {}

        for pos in positions:
            if hasattr(pos, 'symbol') and hasattr(pos, 'size') and hasattr(pos, 'price'):
                symbol = pos.symbol
                size = abs(float(pos.size))
                price = float(pos.price)
                exposure = Decimal(str(size * price))
                total_exposure += exposure

                position_summaries[symbol] = {
                    "size": size,
                    "price": price,
                    "exposure": float(exposure)
                }

        # Calculate config hash for quick comparison
        # Simple hash of sorted key-value pairs (exclude timestamps/metadata)
        config_hash_items = []
        for k, v in sorted(config_dict.items()):
            if k not in {"metadata"} and not isinstance(v, dict):
                config_hash_items.append(f"{k}:{v}")
        config_hash = hash(tuple(config_hash_items))

        # Safely capture environment state
        env_keys = set()
        critical_env_values = {}

        critical_env_vars = {
            "COINBASE_DEFAULT_QUOTE",
            "PERPS_ENABLE_STREAMING",
            "ORDER_PREVIEW_ENABLED"
        }

        for var in critical_env_vars:
            if os.getenv(var) is not None:
                env_keys.add(var)
                # Don't store sensitive values
                if not var.upper().endswith(('_KEY', '_SECRET', '_TOKEN')):
                    critical_env_values[var] = os.getenv(var, "")

        risk_limits = {
            "max_position_size": config_dict.get("max_position_size", "1000"),
            "max_leverage": config_dict.get("max_leverage", 3),
            "daily_loss_limit": config_dict.get("daily_loss_limit", "0")
        }

        return BaselineSnapshot(
            timestamp=datetime.now(UTC),
            config_dict=config_dict.copy(),
            config_hash=str(config_hash),
            env_keys=env_keys,
            critical_env_values=critical_env_values,
            active_symbols=active_symbols.copy(),
            open_positions=position_summaries,
            account_equity=account_equity,
            total_exposure=total_exposure,
            profile=profile,
            broker_type=broker_type,
            risk_limits=risk_limits
        )
