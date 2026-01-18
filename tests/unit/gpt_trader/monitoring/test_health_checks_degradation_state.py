"""Tests for the degradation-state health check."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.monitoring.health_checks import check_degradation_state


class TestCheckDegradationState:
    """Tests for check_degradation_state function."""

    def test_normal_operation(self) -> None:
        """Test healthy state when no degradation."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        healthy, details = check_degradation_state(degradation_state)

        assert healthy is True
        assert details["global_paused"] is False
        assert details["reduce_only_mode"] is False

    def test_global_paused(self) -> None:
        """Test failure when globally paused."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": True,
            "global_reason": "max_reconnect_attempts",
            "paused_symbols": {},
            "global_remaining_seconds": 300,
        }

        healthy, details = check_degradation_state(degradation_state)

        assert healthy is False
        assert details["global_paused"] is True
        assert details["severity"] == "critical"

    def test_reduce_only_mode(self) -> None:
        """Test warning when in reduce-only mode."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        # Create a mock with explicit spec to avoid auto-creating attributes
        risk_manager = MagicMock()
        risk_manager.is_reduce_only_mode = MagicMock(return_value=True)
        risk_manager._reduce_only_mode = True
        risk_manager._reduce_only_reason = "validation_failures"
        # Prevent auto-creation of _cfm_reduce_only_mode attribute
        del risk_manager._cfm_reduce_only_mode
        risk_manager.is_cfm_reduce_only_mode = MagicMock(return_value=False)

        healthy, details = check_degradation_state(degradation_state, risk_manager)

        assert healthy is True  # Reduce-only is warning, not failure
        assert details["reduce_only_mode"] is True
        assert details["reduce_only_reason"] == "validation_failures"
        assert details["severity"] == "warning"

    def test_symbol_paused(self) -> None:
        """Test warning when specific symbols are paused."""
        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {"BTC-USD": {"reason": "rate_limited"}},
            "global_remaining_seconds": 0,
        }

        healthy, details = check_degradation_state(degradation_state)

        assert healthy is True  # Symbol pause is warning, not failure
        assert details["paused_symbol_count"] == 1
        assert "BTC-USD" in details["paused_symbols"]
        assert details["severity"] == "warning"
