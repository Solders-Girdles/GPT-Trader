"""Tests for default runtime guard manager creation and guard state transitions."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.monitoring.guards.base import GuardConfig, GuardStatus
from gpt_trader.monitoring.guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    PositionStuckGuard,
)
from gpt_trader.monitoring.guards.manager import create_default_runtime_guard_manager


class TestDefaultGuardManagerCreation:
    """Test creation of default guard manager."""

    def test_create_default_runtime_guard_manager(self):
        """Test creation of default guard manager with config."""
        config = {
            "circuit_breakers": {
                "daily_loss_limit": 500.0,
                "stale_mark_seconds": 10.0,
                "error_threshold": 5.0,
                "position_timeout_seconds": 900.0,
            },
            "risk_management": {
                "max_drawdown_pct": 3.0,
            },
        }

        manager = create_default_runtime_guard_manager(config)

        # Check all expected guards are present
        expected_guards = {
            "daily_loss",
            "stale_mark",
            "error_rate",
            "position_stuck",
            "max_drawdown",
        }
        assert set(manager.guards.keys()) == expected_guards

        # Check configurations
        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0
        assert daily_loss_guard.config.auto_shutdown is True

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 10.0

        error_rate_guard = manager.guards["error_rate"]
        assert error_rate_guard.config.threshold == 5.0
        assert error_rate_guard.config.auto_shutdown is True

        position_stuck_guard = manager.guards["position_stuck"]
        assert position_stuck_guard.config.threshold == 900.0

        drawdown_guard = manager.guards["max_drawdown"]
        assert drawdown_guard.config.threshold == 3.0
        assert drawdown_guard.config.auto_shutdown is True

    def test_create_default_with_missing_config(self):
        """Test default creation with missing config values."""
        config = {}  # Empty config

        manager = create_default_runtime_guard_manager(config)

        # Should use defaults
        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0  # Default value

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 15.0  # Default value


class TestGuardStateTransitions:
    """Test guard state transitions and status tracking."""

    def test_guard_status_transitions_healthy_to_warning(self):
        """Test guard transitions from healthy to warning."""
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Start healthy
        assert guard.status == GuardStatus.HEALTHY

        # Trigger warning (50% of threshold)
        result = guard.check({"pnl": -40.0})  # -40 is less than -50 (50% of 100)
        assert result is None  # No alert yet
        assert guard.status == GuardStatus.HEALTHY  # Still healthy

        # Trigger warning threshold
        result = guard.check({"pnl": -60.0})  # -60 is more than -50
        assert result is None  # No alert, but status changes
        assert guard.status == GuardStatus.WARNING

    def test_guard_status_transitions_warning_to_breached(self):
        """Test guard transitions from warning to breached."""
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Set up warning state
        guard.check({"pnl": -60.0})
        assert guard.status == GuardStatus.WARNING

        # Breach threshold
        result = guard.check({"pnl": -50.0})  # Total: -110
        assert result is not None
        assert guard.status == GuardStatus.BREACHED
        assert guard.breach_count == 1

    def test_guard_status_transitions_breached_to_warning(self, frozen_time):
        """Test guard transitions from breached back to warning."""
        from datetime import timedelta

        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        # Breach
        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

        # Next day
        frozen_time.tick(delta=timedelta(days=1))
        result = guard.check({"pnl": -10.0})  # New day, small loss
        assert result is None
        assert guard.status == GuardStatus.HEALTHY  # Reset to HEALTHY

        # Breach again to verify reset worked
        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

    def test_position_stuck_guard_state_tracking(self, frozen_time):
        """Test position stuck guard tracks position state."""
        from datetime import timedelta

        guard = PositionStuckGuard(GuardConfig(name="position_stuck", threshold=60.0))  # 1 minute

        # No positions
        result = guard.check({"positions": {}})
        assert result is None

        # Add position
        positions = {"BTC-USD": {"quantity": 1.0}}
        result = guard.check({"positions": positions})
        assert result is None

        # Position still open after timeout
        frozen_time.tick(delta=timedelta(minutes=2))
        result = guard.check({"positions": positions})

        assert result is not None
        assert "Stuck positions detected" in result.message

        # Position closed
        result = guard.check({"positions": {"BTC-USD": {"quantity": 0.0}}})
        assert result is None

    def test_drawdown_guard_peak_tracking(self):
        """Test drawdown guard tracks equity peaks."""
        guard = DrawdownGuard(GuardConfig(name="drawdown", threshold=10.0))

        # Initial equity
        guard.check({"equity": Decimal("1000")})
        assert guard.peak_equity == Decimal("1000")

        # Higher equity
        guard.check({"equity": Decimal("1100")})
        assert guard.peak_equity == Decimal("1100")

        # Drawdown but not breached
        guard.check({"equity": Decimal("1050")})  # 4.55% drawdown
        assert guard.current_drawdown == pytest.approx(Decimal("4.545454545454545"))

        # Breach threshold
        result = guard.check({"equity": Decimal("980")})  # 10.91% drawdown
        assert result is not None
        assert "Maximum drawdown breached" in result.message
