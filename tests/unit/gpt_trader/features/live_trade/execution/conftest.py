from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker with execution defaults."""
    broker = MagicMock()
    broker.place_order = MagicMock()
    broker.list_balances.return_value = []
    broker.list_positions.return_value = []
    broker.cancel_order.return_value = True
    broker.list_orders = None
    broker.get_market_snapshot.return_value = None
    broker.get_product.return_value = None
    return broker


@pytest.fixture
def mock_event_store() -> MagicMock:
    """Create a mock event store."""
    store = MagicMock()
    store.store_event = MagicMock()
    store.append_trade = MagicMock()
    store.append_error = MagicMock()
    return store


@pytest.fixture
def mock_risk_manager() -> MagicMock:
    """Create a mock risk manager with default guard settings."""
    rm = MagicMock()
    rm.track_daily_pnl.return_value = False
    rm.last_mark_update = {}
    rm.config = MagicMock()
    rm.config.volatility_window_periods = 20
    rm.config.slippage_guard_bps = 100
    rm.check_mark_staleness.return_value = False
    rm.is_reduce_only_mode.return_value = False
    rm.pre_trade_validate = MagicMock()
    rm.append_risk_metrics = MagicMock()
    return rm


@pytest.fixture
def mock_equity_calculator() -> MagicMock:
    """Create a mock equity calculator returning a stable equity tuple."""
    return MagicMock(return_value=(Decimal("1000"), [], Decimal("1000")))


@pytest.fixture
def mock_product() -> Product:
    """Create a mock product."""
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=20,
    )


@pytest.fixture
def mock_failure_tracker() -> MagicMock:
    """Create a mock failure tracker."""
    return MagicMock()


@pytest.fixture
def open_orders() -> list[str]:
    """Create an open orders list."""
    return []


@pytest.fixture
def submitter(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
) -> OrderSubmitter:
    """Create an OrderSubmitter instance."""
    return OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot-123",
        open_orders=open_orders,
        integration_mode=False,
    )


@pytest.fixture
def mock_order() -> Order:
    """Create a mock successful order."""
    return Order(
        id="order-123",
        client_id="client-123",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.PENDING,
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
