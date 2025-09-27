"""Behavioral contract tests for broker implementations.

These tests ensure that all broker implementations (mock, Coinbase, etc.)
follow the same behavioral contract. This prevents mocks from diverging
from real broker behavior.
"""

from decimal import Decimal
from datetime import datetime
import pytest
from typing import Protocol, List, Optional
from abc import abstractmethod

from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, Quote, Product, MarketType
)
from bot_v2.orchestration.mock_broker import MockBroker


class BrokerContract(Protocol):
    """Contract that all brokers must satisfy."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to broker."""
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        **kwargs
    ) -> Order:
        """Place an order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        pass
    
    @abstractmethod
    def list_orders(self) -> List[Order]:
        """List all orders."""
        pass
    
    @abstractmethod
    def list_positions(self) -> List[Position]:
        """List all positions."""
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol."""
        pass


class BrokerContractTests:
    """Base tests that all broker implementations must pass."""
    
    @pytest.fixture
    def broker(self) -> IBrokerage:
        """Override this to provide the broker implementation to test."""
        raise NotImplementedError("Subclasses must provide a broker fixture")
    
    def test_connect_succeeds(self, broker):
        """Test that connection can be established."""
        broker.connect()
        # Should not raise
    
    def test_get_quote_has_required_fields(self, broker):
        """Test that quotes have all required fields."""
        broker.connect()
        quote = broker.get_quote("BTC-PERP")
        
        assert quote.symbol == "BTC-PERP"
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.bid < quote.ask  # Spread should be positive
        assert quote.last > 0
        assert quote.ts is not None  # Timestamp field is 'ts'
    
    def test_place_market_order_returns_order(self, broker):
        """Test that placing a market order returns a valid order."""
        broker.connect()
        
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.001")
        )
        
        assert order is not None
        assert order.id is not None
        assert order.symbol == "BTC-PERP"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.qty == Decimal("0.001")
        assert order.status in [OrderStatus.SUBMITTED, OrderStatus.FILLED]
    
    def test_place_limit_order_returns_order(self, broker):
        """Test that placing a limit order returns a valid order."""
        broker.connect()
        quote = broker.get_quote("BTC-PERP")
        
        # Place a buy limit below market
        limit_price = quote.bid * Decimal("0.95")
        
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.001"),
            price=limit_price
        )
        
        assert order is not None
        assert order.id is not None
        assert order.type == OrderType.LIMIT
        assert order.price == limit_price
        assert order.status == OrderStatus.SUBMITTED  # Should not fill immediately
    
    def test_cancel_submitted_order_succeeds(self, broker):
        """Test that canceling a submitted order works."""
        broker.connect()
        quote = broker.get_quote("BTC-PERP")
        
        # Place a limit order far from market
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.001"),
            price=quote.bid * Decimal("0.5")  # Very low price
        )
        
        assert order.status == OrderStatus.SUBMITTED
        
        # Cancel it
        success = broker.cancel_order(order.id)
        assert success is True
        
        # Verify it's cancelled
        updated_order = broker.get_order(order.id)
        assert updated_order.status == OrderStatus.CANCELLED
    
    def test_cancel_nonexistent_order_returns_false(self, broker):
        """Test that canceling a non-existent order returns False."""
        broker.connect()
        
        success = broker.cancel_order("nonexistent_order_id")
        assert success is False
    
    def test_get_order_returns_none_for_unknown(self, broker):
        """Test that get_order returns None for unknown orders."""
        broker.connect()
        
        order = broker.get_order("unknown_order_id")
        # Contract: should return None for unknown orders
        assert order is None
    
    def test_list_orders_includes_placed_orders(self, broker):
        """Test that list_orders includes orders we've placed."""
        broker.connect()
        
        # Place an order
        order = broker.place_order(
            symbol="ETH-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("0.01")
        )
        
        # List should include it
        orders = broker.list_orders()
        order_ids = [o.id for o in orders]
        assert order.id in order_ids
    
    def test_market_order_creates_position(self, broker):
        """Test that filled market orders create positions."""
        broker.connect()
        
        # Check initial positions
        initial_positions = broker.list_positions()
        btc_positions = [p for p in initial_positions if p.symbol == "BTC-PERP"]
        initial_qty = sum(p.qty for p in btc_positions) if btc_positions else Decimal("0")
        
        # Place a market order
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.01")
        )
        
        # Should create or update position
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) > 0
        
        # Total quantity should have increased
        total_qty = sum(p.qty for p in btc_positions)
        assert total_qty > initial_qty
    
    def test_opposite_orders_reduce_position(self, broker):
        """Test that opposite orders reduce positions."""
        broker.connect()
        
        # Open a long position
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.02")
        )
        
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) > 0
        initial_qty = btc_positions[0].qty
        
        # Reduce with opposite order
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("0.01")
        )
        
        # Position should be reduced
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        if btc_positions:  # Might be closed completely
            assert btc_positions[0].qty < initial_qty


class TestMockBrokerContract(BrokerContractTests):
    """Test that MockBroker satisfies the broker contract."""
    
    @pytest.fixture
    def broker(self):
        """Provide MockBroker for testing."""
        return MockBroker()


class TestBrokerBehaviorConsistency:
    """Test consistency of broker behaviors."""
    
    def test_order_status_transitions_are_valid(self):
        """Test that order status transitions follow valid paths."""
        broker = MockBroker()
        broker.connect()
        
        # Place a limit order
        quote = broker.get_quote("BTC-PERP")
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.001"),
            price=quote.bid * Decimal("0.95")
        )
        
        initial_status = order.status
        assert initial_status == OrderStatus.SUBMITTED
        
        # Cancel it
        broker.cancel_order(order.id)
        cancelled_order = broker.get_order(order.id)
        
        # Valid transition: SUBMITTED -> CANCELLED
        assert cancelled_order.status == OrderStatus.CANCELLED
    
    def test_filled_orders_cannot_be_cancelled(self):
        """Test that filled orders cannot be cancelled."""
        broker = MockBroker()
        broker.connect()
        
        # Place a market order (should fill immediately)
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.001")
        )
        
        # Try to cancel (should fail if filled)
        if order.status == OrderStatus.FILLED:
            success = broker.cancel_order(order.id)
            assert success is False
    
    def test_position_math_is_consistent(self):
        """Test that position calculations are mathematically consistent."""
        broker = MockBroker()
        broker.connect()
        
        # Start with no position
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) == 0
        
        # Buy 0.03
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.03")
        )
        
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) == 1
        assert btc_positions[0].qty == Decimal("0.03")
        
        # Sell 0.01
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("0.01")
        )
        
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) == 1
        assert btc_positions[0].qty == Decimal("0.02")
        
        # Sell 0.02 (should close position)
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("0.02")
        )
        
        positions = broker.list_positions()
        btc_positions = [p for p in positions if p.symbol == "BTC-PERP"]
        assert len(btc_positions) == 0 or btc_positions[0].qty == Decimal("0")
