"""Tests for StateValidator."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.state_management.validators import StateValidator, ValidationResult


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_initial_state_is_valid(self):
        """Test new ValidationResult is valid with empty lists."""
        result = ValidationResult()

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_sets_invalid(self):
        """Test adding error marks result as invalid."""
        result = ValidationResult()
        result.add_error("field", "error message", "value")

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "field"
        assert result.errors[0].message == "error message"
        assert result.errors[0].severity == "error"

    def test_add_warning_keeps_valid(self):
        """Test adding warning does not invalidate result."""
        result = ValidationResult()
        result.add_warning("field", "warning message")

        assert result.valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"

    def test_merge_combines_results(self):
        """Test merge combines errors and warnings from both results."""
        result1 = ValidationResult()
        result1.add_error("field1", "error1")

        result2 = ValidationResult()
        result2.add_warning("field2", "warning2")

        result1.merge(result2)

        assert result1.valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1

    def test_merge_propagates_invalid_state(self):
        """Test merging invalid result makes target invalid."""
        result1 = ValidationResult()  # valid
        result2 = ValidationResult()
        result2.add_error("field", "error")  # invalid

        result1.merge(result2)

        assert result1.valid is False


class TestStateValidator:
    """Test StateValidator functionality."""

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

    def test_validate_market_valid_prices(self):
        """Test market validation passes for valid prices."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "50000.00", "ETH-USD": "3000.00"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert result.valid

    def test_validate_market_negative_price_error(self):
        """Test market validation catches negative prices."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "-100.00"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert not result.valid
        assert any("Negative price" in e.message for e in result.errors)

    def test_validate_market_invalid_price_format(self):
        """Test market validation catches invalid price format."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "not_a_number"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert not result.valid
        assert any("Invalid price format" in e.message for e in result.errors)

    def test_validate_orders_valid(self):
        """Test order validation passes for valid orders."""
        validator = StateValidator()
        mock_order = MagicMock()
        mock_order.order_id = "order-123"
        mock_order.symbol = "BTC-USD"
        mock_order.side = "BUY"
        mock_order.quantity = "1.5"
        mock_order.price = "50000.00"

        result = validator._validate_orders([mock_order])

        assert result.valid

    def test_validate_orders_missing_order_id(self):
        """Test order validation warns on missing order_id."""
        validator = StateValidator()
        mock_order = MagicMock()
        mock_order.order_id = None
        mock_order.symbol = "BTC-USD"
        mock_order.side = "BUY"

        result = validator._validate_orders([mock_order])

        assert len(result.warnings) > 0

    def test_validate_trades_valid(self):
        """Test trade validation passes for valid trades."""
        validator = StateValidator()
        mock_trade = MagicMock()
        mock_trade.trade_id = "trade-123"
        mock_trade.symbol = "BTC-USD"
        mock_trade.side = "BUY"
        mock_trade.quantity = "1.0"
        mock_trade.price = "50000.00"
        mock_trade.fee = "10.00"

        result = validator._validate_trades([mock_trade])

        assert result.valid

    def test_validate_trades_negative_quantity_error(self):
        """Test trade validation catches negative quantity."""
        validator = StateValidator()
        mock_trade = MagicMock()
        mock_trade.trade_id = "trade-123"
        mock_trade.symbol = "BTC-USD"
        mock_trade.quantity = "-1.0"
        mock_trade.price = "50000.00"
        mock_trade.fee = "0"

        result = validator._validate_trades([mock_trade])

        assert not result.valid

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

    def test_validate_price_helper(self):
        """Test _validate_price helper function."""
        validator = StateValidator()

        # Valid price
        result = validator._validate_price("50000.00", "test.price")
        assert result.valid

        # Negative price
        result = validator._validate_price("-100", "test.price")
        assert not result.valid

        # Invalid format
        result = validator._validate_price("abc", "test.price")
        assert not result.valid

    def test_validate_quantity_helper(self):
        """Test _validate_quantity helper function."""
        validator = StateValidator()

        # Valid positive quantity
        result = validator._validate_quantity("100", "test.quantity")
        assert result.valid

        # Valid negative (for shorts)
        result = validator._validate_quantity("-100", "test.quantity", allow_negative=True)
        assert result.valid

        # Invalid negative when not allowed
        result = validator._validate_quantity("-100", "test.quantity", allow_negative=False)
        assert not result.valid
