"""Tests for default runtime guard manager creation."""

from __future__ import annotations

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
