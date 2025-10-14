"""Tests for runtime configuration drift detection."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from bot_v2.config.types import Profile
from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.monitoring.configuration_guardian import (
    BaselineSnapshot,
    ConfigurationGuardian,
    DriftEvent,
    EnvironmentMonitor,
)


class TestBaselineSnapshot:
    """Test baseline snapshot creation."""

    def test_create_baseline_snapshot(self):
        """Test creating a baseline snapshot."""
        config_dict = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "max_leverage": 3,
            "mock_broker": True,
        }

        snapshot = ConfigurationGuardian.create_baseline_snapshot(
            config_dict=config_dict,
            active_symbols=["BTC-USD"],
            positions=[],
            account_equity=Decimal("10000"),
            profile=Profile.DEV,
            broker_type="mock",
        )

        assert snapshot.profile == Profile.DEV
        assert snapshot.active_symbols == ["BTC-USD"]
        assert snapshot.account_equity == Decimal("10000")
        assert snapshot.broker_type == "mock"
        assert snapshot.total_exposure == Decimal("0")  # No positions


class TestEnvironmentMonitor:
    """Test environment variable monitoring."""

    def create_baseline_snapshot(self):
        """Helper to create test baseline."""
        return BaselineSnapshot(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            config_dict={},
            config_hash="",
            env_keys=set(),
            critical_env_values={},
            active_symbols=[],
            open_positions={},
            account_equity=None,
            total_exposure=Decimal("0"),
            profile=Profile.DEV,
            broker_type="mock",
            risk_limits={},
        )

    def test_no_env_changes(self):
        """Test no environment changes."""
        with patch.dict("os.environ", {}, clear=True):
            settings = load_runtime_settings()
            baseline = self.create_baseline_snapshot()
            monitor = EnvironmentMonitor(baseline, settings=settings)
            events = monitor.check_changes()
        assert len(events) == 0

    def test_critical_env_change(self):
        """Test critical environment variable change."""
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "1"}, clear=True):
            settings = load_runtime_settings()
            baseline = self.create_baseline_snapshot()
            baseline.env_keys.add("COINBASE_ENABLE_DERIVATIVES")
            monitor = EnvironmentMonitor(baseline, settings=settings)

            with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "0"}, clear=True):
                monitor._settings = load_runtime_settings()
                events = monitor.check_changes()

        assert len(events) == 1
        assert events[0].drift_type == "critical_env_changed"
        assert events[0].severity == "critical"
        assert events[0].applied_response == "emergency_shutdown"

    def test_risk_env_change(self):
        """Test risk environment variable change."""
        with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.1"}, clear=True):
            settings = load_runtime_settings()
            baseline = self.create_baseline_snapshot()
            baseline.env_keys.add("PERPS_POSITION_FRACTION")
            monitor = EnvironmentMonitor(baseline, settings=settings)

            with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.2"}, clear=True):
                monitor._settings = load_runtime_settings()
                events = monitor.check_changes()

        assert len(events) == 1
        assert events[0].drift_type == "risk_env_changed"
        assert events[0].severity == "high"
        assert events[0].applied_response == "reduce_only"

    def test_monitored_env_change(self):
        """Test monitored environment variable change."""
        with patch.dict("os.environ", {"COINBASE_DEFAULT_QUOTE": "USD"}, clear=True):
            settings = load_runtime_settings()
            baseline = self.create_baseline_snapshot()
            baseline.env_keys.add("COINBASE_DEFAULT_QUOTE")
            monitor = EnvironmentMonitor(baseline, settings=settings)

            with patch.dict("os.environ", {"COINBASE_DEFAULT_QUOTE": "EUR"}, clear=True):
                monitor._settings = load_runtime_settings()
                events = monitor.check_changes()

        assert len(events) == 1
        assert events[0].drift_type == "monitored_env_changed"
        assert events[0].severity == "low"
        assert events[0].applied_response == "sticky"


class TestConfigurationGuardian:
    """Test the main guardian class."""

    def create_test_guardian(self):
        """Create a test guardian."""
        baseline = BaselineSnapshot(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            config_dict={"symbols": ["BTC-USD"], "max_leverage": 3, "profile": Profile.DEV},
            config_hash="",
            env_keys=set(),
            critical_env_values={},
            active_symbols=["BTC-USD"],
            open_positions={},
            account_equity=Decimal("10000"),
            total_exposure=Decimal("0"),
            profile=Profile.DEV,
            broker_type="mock",
            risk_limits={},
        )

        return ConfigurationGuardian(baseline, settings=load_runtime_settings())

    @patch.dict("os.environ", {}, clear=True)
    def test_guardian_initialization(self):
        """Test guardian initialization."""
        guardian = self.create_test_guardian()
        status = guardian.get_health_status()

        assert status["monitors_status"]["environment_monitor"] == "healthy"
        assert status["drift_summary"]["total_events"] == 0

    def test_pre_cycle_check_clean(self):
        """Test pre-cycle check with no issues."""
        with patch.dict("os.environ", {}, clear=True):
            guardian = self.create_test_guardian()
            result = guardian.pre_cycle_check()

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_pre_cycle_check_env_drift(self):
        """Test pre-cycle check with environment drift."""
        # Start with the env var set at baseline
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "1"}, clear=True):
            guardian = self.create_test_guardian()
            # Set baseline to see the env var as existing
            guardian.environment_monitor.baseline.env_keys.add("COINBASE_ENABLE_DERIVATIVES")
            guardian.environment_monitor._settings = load_runtime_settings()

        # Now change it and check
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "0"}, clear=True):
            guardian.environment_monitor._settings = load_runtime_settings()
            result = guardian.pre_cycle_check()

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("critical_env_changed" in error for error in result.errors)

    def test_pre_cycle_check_high_severity_env_drift(self):
        """Test pre-cycle check fails validation for high-severity risks."""
        # Start with the env var set at baseline
        with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.1"}, clear=True):
            guardian = self.create_test_guardian()
            # Set baseline to see the env var as existing
            guardian.environment_monitor.baseline.env_keys.add("PERPS_POSITION_FRACTION")
            guardian.environment_monitor._settings = load_runtime_settings()

        # Change it to a different value (high-severity event)
        with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.3"}, clear=True):
            guardian.environment_monitor._settings = load_runtime_settings()
            result = guardian.pre_cycle_check()

        assert result.is_valid is False  # High-severity should fail validation
        assert len(result.errors) > 0
        assert any("risk_env_changed" in error for error in result.errors)

    def test_state_validator_leverage_violation(self):
        """Test state validator for leverage violations."""
        baseline = self.create_test_guardian().baseline

        # Example: Create a large position that violates position size limits
        mock_position = MagicMock()
        mock_position.size = Decimal("0.1")  # Mock doesn't need spec
        mock_position.price = Decimal("50000")  # $5000 position

        result = baseline.validate_config_against_state(
            {"max_position_size": Decimal("4000")},  # New limit below current position
            [],
            [mock_position],
            Decimal("10000"),
        )

        assert len(result) > 0
        assert any("position_size_violation_current_exposure" in e.drift_type for e in result)

    def test_state_validator_symbol_removal_violation(self):
        """Test state validator when removing symbols with positions."""
        baseline = self.create_test_guardian().baseline

        # Create a position for BTC-USD
        mock_position = MagicMock(spec=Position)
        mock_position.symbol = "BTC-USD"

        # Try to remove BTC-USD from symbols
        result = baseline.validate_config_against_state(
            {"symbols": ["ETH-USD"]}, [], [mock_position], Decimal("10000")  # Remove BTC-USD
        )

        assert len(result) > 0
        assert any("symbols_remove_active_positions" in e.drift_type for e in result)

    def test_state_validator_profile_change_violation(self):
        """Test state validator when changing profile."""
        baseline = self.create_test_guardian().baseline

        # Try to change profile during runtime
        result = baseline.validate_config_against_state(
            {"profile": Profile.CANARY}, [], [], Decimal("10000")  # Change from DEV to CANARY
        )

        assert len(result) > 0
        assert any("profile_changed_during_runtime" in e.drift_type for e in result)

    def test_critical_events_detection(self):
        """Test that critical events are properly detected for emergency shutdown."""
        # Start with the env var set at baseline
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "1"}, clear=True):
            guardian = self.create_test_guardian()
            # Set baseline to see the env var as existing
            guardian.environment_monitor.baseline.env_keys.add("COINBASE_ENABLE_DERIVATIVES")
            guardian.environment_monitor._settings = load_runtime_settings()

        # Change it to a different value (critical event)
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "0"}, clear=True):
            guardian.environment_monitor._settings = load_runtime_settings()
            result = guardian.pre_cycle_check()

        assert result.is_valid is False

        # Most importantly - result contains critical event indication
        critical_errors = [
            error
            for error in result.errors
            if any(keyword in error.lower() for keyword in ["critical", "emergency_shutdown"])
        ]
        assert len(critical_errors) > 0, f"Expected critical errors but got only: {result.errors}"
