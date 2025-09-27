"""
Unit tests for pre-flight order validation in execution engine.

Tests validation violations and successful cases with adjusted values.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine
from bot_v2.features.brokerages.core.interfaces import (
    OrderType, OrderSide, TimeInForce, Quote
)


class TestPreflightValidation:
    """Test pre-flight validation in execution engine."""
    
    @pytest.fixture
    def mock_broker(self):
        """Create mock broker with required methods."""
        broker = Mock()
        broker.get_product = Mock()
        broker.get_quote = Mock()
        broker.place_order = Mock()
        return broker
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        risk_manager = Mock()
        risk_manager.pre_trade_validate = Mock()
        return risk_manager
    
    @pytest.fixture
    def engine(self, mock_broker, mock_risk_manager):
        """Create execution engine with mocks."""
        return AdvancedExecutionEngine(
            broker=mock_broker,
            risk_manager=mock_risk_manager
        )
    
    def test_validation_rejects_below_min_size(self, engine, mock_broker):
        """Test order rejected when size below minimum."""
        # Use real Product (fewer mocks)
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=None,
            price_increment=Decimal("0.01"),
        )
        mock_broker.get_product.return_value = product
        
        # Attempt to place order below min_size
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.005"),  # Below min of 0.01
            order_type=OrderType.MARKET
        )
        
        # Should be rejected
        assert order is None
        assert engine.order_metrics['rejected'] > 0
        assert 'min_size' in engine.rejections_by_reason
    
    def test_validation_rejects_below_min_notional(self, engine, mock_broker):
        """Test order rejected when notional below minimum."""
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("100"),
            price_increment=Decimal("0.01"),
        )
        mock_broker.get_product.return_value = product
        
        # Attempt limit order with low notional
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10")  # Notional = 0.001 * 10 = 0.01, way below 100
        )
        
        # Should be rejected
        assert order is None
        assert engine.order_metrics['rejected'] > 0
        assert 'min_notional' in engine.rejections_by_reason
    
    def test_validation_adjusts_size_to_step(self, engine, mock_broker):
        """Test size is adjusted to step size."""
        # Use real Product to reduce mocking
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=None,
            price_increment=Decimal("0.01"),
        )
        mock_broker.get_product.return_value = product
        
        # Mock successful order placement
        mock_order = Mock()
        mock_order.id = "order-123"
        mock_broker.place_order.return_value = mock_order
        
        # Place order with unaligned size
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1234"),  # Will be floored to 0.123
            order_type=OrderType.MARKET
        )
        
        # Check broker was called with adjusted size
        mock_broker.place_order.assert_called_once()
        call_args = mock_broker.place_order.call_args
        assert call_args.kwargs['qty'] == Decimal("0.123")
    
    def test_validation_adjusts_price_side_aware(self, engine, mock_broker):
        """Test price is adjusted based on side."""
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=None,
            price_increment=Decimal("0.01"),
        )
        mock_broker.get_product.return_value = product
        
        # Mock successful order placement
        mock_order = Mock()
        mock_order.id = "order-123"
        mock_broker.place_order.return_value = mock_order
        
        # Test BUY order (should floor price)
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50123.456")  # Will be floored to 50123.45
        )
        
        call_args = mock_broker.place_order.call_args
        assert call_args.kwargs['price'] == Decimal("50123.45")
        
        # Reset mock
        mock_broker.place_order.reset_mock()
        
        # Test SELL order (should ceil price)
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50123.456")  # Will be ceiled to 50123.46
        )
        
        call_args = mock_broker.place_order.call_args
        assert call_args.kwargs['price'] == Decimal("50123.46")
    
    def test_validation_handles_xrp_ten_units(self, engine, mock_broker):
        """Test XRP-PERP respects 10-unit step size."""
        from bot_v2.features.brokerages.core.interfaces import Product, MarketType
        product = Product(
            symbol="XRP-PERP",
            base_asset="XRP",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("10"),
            step_size=Decimal("10"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.0001"),
        )
        mock_broker.get_product.return_value = product
        
        # Mock successful order
        mock_order = Mock()
        mock_order.id = "order-xrp"
        mock_broker.place_order.return_value = mock_order
        
        # Place order with unaligned XRP quantity
        order = engine.place_order(
            symbol="XRP-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("125"),  # Will be floored to 120 (10-unit steps)
            order_type=OrderType.MARKET
        )
        
        # Check adjusted to 10-unit boundary
        call_args = mock_broker.place_order.call_args
        assert call_args.kwargs['qty'] == Decimal("120")
    
    def test_validation_passes_valid_order(self, engine, mock_broker):
        """Test valid order passes validation."""
        # Setup mock product
        mock_product = Mock()
        mock_product.step_size = Decimal("0.001")
        mock_product.min_size = Decimal("0.001")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_notional = Decimal("10")
        mock_broker.get_product.return_value = mock_product
        
        # Mock successful order
        mock_order = Mock()
        mock_order.id = "order-valid"
        mock_broker.place_order.return_value = mock_order
        
        # Place valid order
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000")  # Notional = 0.1 * 50000 = 5000, well above min
        )
        
        # Should succeed
        assert order is not None
        assert order.id == "order-valid"
        assert engine.order_metrics['placed'] > 0
        assert engine.order_metrics['rejected'] == 0
    
    def test_validation_tracks_rejection_reasons(self, engine, mock_broker):
        """Test rejection reasons are tracked properly."""
        # Setup mock product
        mock_product = Mock()
        mock_product.step_size = Decimal("0.001")
        mock_product.min_size = Decimal("0.01")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_notional = Decimal("100")
        mock_broker.get_product.return_value = mock_product
        
        # Test multiple rejection scenarios
        
        # 1. Below min_size
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.005"),
            order_type=OrderType.MARKET
        )
        assert engine.rejections_by_reason.get('min_size', 0) == 1
        
        # 2. Below min_notional
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("1")  # Notional = 0.01 * 1 = 0.01, below 100
        )
        assert engine.rejections_by_reason.get('min_notional', 0) == 1
        
        # Check total rejections
        assert engine.order_metrics['rejected'] == 2


class TestRiskIntegration:
    """Test risk manager integration with pre-flight checks."""
    
    def test_risk_check_called_before_spec_validation(self):
        """Test risk manager is called during pre-flight."""
        mock_broker = Mock()
        mock_risk_manager = Mock()
        
        # Setup mock product
        mock_product = Mock()
        mock_product.step_size = Decimal("0.001")
        mock_product.min_size = Decimal("0.001")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_notional = None
        mock_broker.get_product.return_value = mock_product
        
        # Risk manager should raise validation error
        from bot_v2.features.live_trade.risk import ValidationError
        mock_risk_manager.pre_trade_validate.side_effect = ValidationError("Risk limit exceeded")
        
        engine = AdvancedExecutionEngine(
            broker=mock_broker,
            risk_manager=mock_risk_manager
        )
        
        # Place order
        order = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET
        )
        
        # Should be rejected by risk
        assert order is None
        assert mock_risk_manager.pre_trade_validate.called
        assert engine.order_metrics['rejected'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
