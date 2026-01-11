from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import MarketType, Product


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
