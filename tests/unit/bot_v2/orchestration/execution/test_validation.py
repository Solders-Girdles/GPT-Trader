"""Tests for order validation"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.risk import ValidationError
from bot_v2.orchestration.execution.validation import OrderValidator


@pytest.fixture
def mock_broker():
    """Mock broker"""
    broker = Mock()
    broker.get_market_snapshot = Mock()
    broker.preview_order = Mock()
    return broker


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager"""
    manager = Mock()
    manager.config = Mock()
    manager.config.slippage_guard_bps = 100
    manager.check_mark_staleness = Mock(return_value=False)
    manager.pre_trade_validate = Mock()
    manager.is_reduce_only_mode = Mock(return_value=False)
    return manager


@pytest.fixture
def mock_product():
    """Mock product"""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD"
    product.base_increment = Decimal("0.00000001")
    product.quote_increment = Decimal("0.01")
    product.price_increment = Decimal("0.01")
    return product


@pytest.fixture
def order_validator(mock_broker, mock_risk_manager):
    """Create OrderValidator instance"""
    record_preview = Mock()
    record_rejection = Mock()
    return OrderValidator(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        enable_order_preview=True,
        record_preview_callback=record_preview,
        record_rejection_callback=record_rejection,
    )


class TestOrderValidator:
    """Test suite for OrderValidator"""

    def test_initialization(self, order_validator):
        """Test validator initialization"""
        assert order_validator.broker is not None
        assert order_validator.risk_manager is not None
        assert order_validator.enable_order_preview is True

    @patch('bot_v2.orchestration.execution.validation.spec_validate_order')
    def test_validate_exchange_rules_success(
        self, mock_spec_validate, order_validator, mock_product
    ):
        """Test successful exchange rules validation"""
        mock_spec_validate.return_value = Mock(
            ok=True, adjusted_quantity=None, adjusted_price=None
        )

        quantity, price = order_validator.validate_exchange_rules(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.01"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        assert quantity == Decimal("0.01")
        assert price is None

    @patch('bot_v2.orchestration.execution.validation.spec_validate_order')
    def test_validate_exchange_rules_with_adjustments(
        self, mock_spec_validate, order_validator, mock_product
    ):
        """Test validation with quantity/price adjustments"""
        mock_spec_validate.return_value = Mock(
            ok=True,
            adjusted_quantity=Decimal("0.02"),
            adjusted_price=Decimal("50000.50"),
        )

        quantity, price = order_validator.validate_exchange_rules(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("0.015"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        assert quantity == Decimal("0.02")
        assert price == Decimal("50000.50")

    @patch('bot_v2.orchestration.execution.validation.spec_validate_order')
    def test_validate_exchange_rules_failure(
        self, mock_spec_validate, order_validator, mock_product
    ):
        """Test validation failure"""
        mock_spec_validate.return_value = Mock(ok=False, reason="quantity_too_small")

        with pytest.raises(ValidationError, match="Spec validation failed"):
            order_validator.validate_exchange_rules(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.0001"),
                price=None,
                effective_price=Decimal("50000"),
                product=mock_product,
            )

        order_validator._record_rejection.assert_called_once()

    def test_ensure_mark_is_fresh_success(self, order_validator, mock_risk_manager):
        """Test mark freshness check passes"""
        mock_risk_manager.check_mark_staleness.return_value = False

        # Should not raise
        order_validator.ensure_mark_is_fresh("BTC-USD")

    def test_ensure_mark_is_fresh_stale(self, order_validator, mock_risk_manager):
        """Test mark freshness check with stale data"""
        mock_risk_manager.check_mark_staleness.return_value = True

        with pytest.raises(ValidationError, match="Mark price is stale"):
            order_validator.ensure_mark_is_fresh("BTC-USD")

    def test_enforce_slippage_guard_no_snapshot(self, order_validator, mock_broker):
        """Test slippage guard without market snapshot"""
        mock_broker.get_market_snapshot.return_value = None

        # Should pass without snapshot data
        order_validator.enforce_slippage_guard(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
        )

    def test_enforce_slippage_guard_within_limit(self, order_validator, mock_broker):
        """Test slippage guard within acceptable limits"""
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 10,
            "depth_l1": 1000000,  # High liquidity
        }

        # Should pass with low slippage
        order_validator.enforce_slippage_guard(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.01"),  # Small order
            effective_price=Decimal("50000"),
        )

    def test_enforce_slippage_guard_exceeds_limit(
        self, order_validator, mock_broker, mock_risk_manager
    ):
        """Test slippage guard exceeds limit"""
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 50,
            "depth_l1": 1000,  # Very low liquidity
        }
        mock_risk_manager.config.slippage_guard_bps = 100

        with pytest.raises(ValidationError, match="Expected slippage"):
            order_validator.enforce_slippage_guard(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_quantity=Decimal("10"),  # Large order
                effective_price=Decimal("50000"),
            )

    def test_run_pre_trade_validation(
        self, order_validator, mock_risk_manager, mock_product
    ):
        """Test pre-trade validation"""
        order_validator.run_pre_trade_validation(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
            product=mock_product,
            equity=Decimal("10000"),
            current_positions={},
        )

        mock_risk_manager.pre_trade_validate.assert_called_once()

    def test_maybe_preview_order_enabled(self, order_validator, mock_broker):
        """Test order preview when enabled"""
        mock_broker.preview_order.return_value = {"estimated_fee": "5.00"}

        order_validator.maybe_preview_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        mock_broker.preview_order.assert_called_once()
        order_validator._record_preview.assert_called_once()

    def test_maybe_preview_order_disabled(self, mock_broker, mock_risk_manager):
        """Test order preview when disabled"""
        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,
            record_preview_callback=Mock(),
            record_rejection_callback=Mock(),
        )

        validator.maybe_preview_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        # Should not call preview
        mock_broker.preview_order.assert_not_called()

    def test_maybe_preview_order_with_tif(self, order_validator, mock_broker):
        """Test order preview with time in force"""
        mock_broker.preview_order.return_value = {}

        order_validator.maybe_preview_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.IOC,
            reduce_only=False,
            leverage=2,
        )

        call_args = mock_broker.preview_order.call_args[1]
        assert call_args["tif"] == TimeInForce.IOC

    def test_maybe_preview_order_exception_handling(
        self, order_validator, mock_broker
    ):
        """Test preview order handles exceptions gracefully"""
        mock_broker.preview_order.side_effect = Exception("Preview failed")

        # Should not raise, just log
        order_validator.maybe_preview_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

    def test_finalize_reduce_only_flag_normal(
        self, order_validator, mock_risk_manager
    ):
        """Test reduce-only flag finalization in normal mode"""
        mock_risk_manager.is_reduce_only_mode.return_value = False

        result = order_validator.finalize_reduce_only_flag(False, "BTC-USD")

        assert result is False

    def test_finalize_reduce_only_flag_forced(
        self, order_validator, mock_risk_manager
    ):
        """Test reduce-only flag finalization when mode is active"""
        mock_risk_manager.is_reduce_only_mode.return_value = True

        result = order_validator.finalize_reduce_only_flag(False, "BTC-USD")

        assert result is True

    def test_finalize_reduce_only_flag_already_true(
        self, order_validator, mock_risk_manager
    ):
        """Test reduce-only flag when already true"""
        mock_risk_manager.is_reduce_only_mode.return_value = False

        result = order_validator.finalize_reduce_only_flag(True, "BTC-USD")

        assert result is True

    @patch('bot_v2.orchestration.execution.validation.spec_validate_order')
    def test_validate_limit_order_price_quantization(
        self, mock_spec_validate, order_validator, mock_product
    ):
        """Test limit order price is quantized"""
        mock_spec_validate.return_value = Mock(
            ok=True, adjusted_quantity=None, adjusted_price=None
        )

        quantity, price = order_validator.validate_exchange_rules(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("0.01"),
            price=Decimal("50000.123"),  # Needs quantization
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        # Price should be quantized to price_increment
        assert price is not None
        # Should be rounded to nearest 0.01

    def test_ensure_mark_is_fresh_exception_handling(
        self, order_validator, mock_risk_manager
    ):
        """Test mark freshness check handles non-ValidationError exceptions"""
        mock_risk_manager.check_mark_staleness.side_effect = RuntimeError("Test error")

        # Should not raise for non-ValidationError
        order_validator.ensure_mark_is_fresh("BTC-USD")

    def test_enforce_slippage_guard_exception_handling(
        self, order_validator, mock_broker
    ):
        """Test slippage guard handles exceptions gracefully"""
        mock_broker.get_market_snapshot.side_effect = Exception("Snapshot failed")

        # Should not raise
        order_validator.enforce_slippage_guard(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
        )