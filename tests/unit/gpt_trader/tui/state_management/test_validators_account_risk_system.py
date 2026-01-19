"""Tests for StateValidator account, risk, and system validation."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.state_management.validators import StateValidator


class TestStateValidatorAccount:
    """Test StateValidator account validation."""

    def test_validate_account_valid(self):
        """Test account validation passes for valid account."""
        validator = StateValidator()
        mock_balance = MagicMock()
        mock_balance.asset = "USD"
        mock_balance.total = Decimal("10000.00")
        mock_balance.available = Decimal("9000.00")

        mock_account = MagicMock()
        mock_account.balances = [mock_balance]

        result = validator._validate_account(mock_account)

        assert result.valid

    def test_validate_account_available_exceeds_total_warning(self):
        """Test account validation warns when available exceeds total."""
        validator = StateValidator()
        mock_balance = MagicMock()
        mock_balance.asset = "USD"
        mock_balance.total = Decimal("1000.00")
        mock_balance.available = Decimal("2000.00")  # More than total

        mock_account = MagicMock()
        mock_account.balances = [mock_balance]

        result = validator._validate_account(mock_account)

        assert len(result.warnings) > 0
        assert any("exceeds total" in w.message for w in result.warnings)


class TestStateValidatorRisk:
    """Test StateValidator risk validation."""

    def test_validate_risk_valid(self):
        """Test risk validation passes for valid risk data."""
        validator = StateValidator()
        mock_risk = MagicMock()
        mock_risk.max_leverage = 10.0
        mock_risk.daily_loss_limit_pct = 5.0
        mock_risk.current_daily_loss_pct = 1.0

        result = validator._validate_risk(mock_risk)

        assert result.valid

    def test_validate_risk_leverage_out_of_range(self):
        """Test risk validation warns on excessive leverage."""
        validator = StateValidator()
        mock_risk = MagicMock()
        mock_risk.max_leverage = 200.0  # Exceeds max
        mock_risk.daily_loss_limit_pct = 5.0
        mock_risk.current_daily_loss_pct = 1.0

        result = validator._validate_risk(mock_risk)

        assert len(result.warnings) > 0


class TestStateValidatorSystem:
    """Test StateValidator system validation."""

    def test_validate_system_valid(self):
        """Test system validation passes for valid system data."""
        validator = StateValidator()
        mock_system = MagicMock()
        mock_system.api_latency = 0.05
        mock_system.connection_status = "CONNECTED"

        result = validator._validate_system(mock_system)

        assert result.valid

    def test_validate_system_unknown_status_warning(self):
        """Test system validation warns on unknown connection status."""
        validator = StateValidator()
        mock_system = MagicMock()
        mock_system.api_latency = 0.05
        mock_system.connection_status = "WEIRD_STATUS"

        result = validator._validate_system(mock_system)

        assert len(result.warnings) > 0
