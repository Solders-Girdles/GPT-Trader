"""Tests for StateValidator full-state validation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.tui.state_management.validators import StateValidator


class TestStateValidatorFullState:
    """Test StateValidator validate_full_state behavior."""

    def test_validate_full_state_handles_none_components(self):
        """Test validation handles None components gracefully."""
        validator = StateValidator()
        mock_status = MagicMock()
        mock_status.market = None
        mock_status.positions = None
        mock_status.orders = None
        mock_status.trades = None
        mock_status.account = None
        mock_status.risk = None
        mock_status.system = None

        result = validator.validate_full_state(mock_status)

        # Should have errors for None components
        assert not result.valid
        assert len(result.errors) > 0
