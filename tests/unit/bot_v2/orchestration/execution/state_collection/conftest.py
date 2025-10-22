"""Shared fixtures for state collection tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, IBrokerage, Product
from bot_v2.orchestration.execution.state_collection import StateCollector
from bot_v2.orchestration.runtime_settings import RuntimeSettings


@pytest.fixture
def mock_brokerage():
    """Mock brokerage adapter for state collection tests."""
    broker = MagicMock(spec=IBrokerage)

    # Mock balance list
    broker.list_balances.return_value = [
        Balance(asset="USD", available=Decimal("10000.0"), total=Decimal("10000.0"), hold=Decimal("0")),
        Balance(asset="BTC", available=Decimal("0.5"), total=Decimal("0.5"), hold=Decimal("0")),
        Balance(asset="USDC", available=Decimal("5000.0"), total=Decimal("5000.0"), hold=Decimal("0")),
    ]

    # Mock position list
    broker.list_positions.return_value = []

    # Mock product methods
    broker.get_product.return_value = MagicMock(
        symbol="BTC-PERP",
        bid_price=Decimal("50000.0"),
        ask_price=Decimal("50010.0"),
        price=Decimal("50005.0"),
        quote_increment=Decimal("0.1")
    )

    # Mock price methods
    broker.get_mark_price = MagicMock(return_value=Decimal("50005.0"))
    broker.get_quote = MagicMock()
    broker.get_quote.return_value = MagicMock(last=Decimal("50005.0"))

    return broker


@pytest.fixture
def mock_runtime_settings():
    """Mock runtime settings for tests."""
    settings = MagicMock(spec=RuntimeSettings)
    settings.raw_env = {
        "PERPS_COLLATERAL_ASSETS": "USD,USDC,ETH"
    }
    return settings


@pytest.fixture
def state_collector(mock_brokerage):
    """StateCollector instance with mocked dependencies."""
    return StateCollector(mock_brokerage)


@pytest.fixture
def state_collector_with_settings(mock_brokerage, mock_runtime_settings):
    """StateCollector instance with custom runtime settings."""
    return StateCollector(mock_brokerage, settings=mock_runtime_settings)


@pytest.fixture
def sample_balances():
    """Sample balance data for testing."""
    return [
        Balance(asset="USD", available=Decimal("15000.0"), total=Decimal("15000.0"), hold=Decimal("0")),
        Balance(asset="USDC", available=Decimal("25000.0"), total=Decimal("25000.0"), hold=Decimal("0")),
        Balance(asset="ETH", available=Decimal("2.0"), total=Decimal("2.0"), hold=Decimal("0")),
        Balance(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1"), hold=Decimal("0")),
    ]


@pytest.fixture
def sample_positions():
    """Sample position data for testing."""
    position1 = MagicMock()
    position1.symbol = "BTC-PERP"
    position1.quantity = Decimal("0.5")
    position1.side = "long"
    position1.entry_price = Decimal("45000.0")
    position1.mark_price = Decimal("50000.0")

    position2 = MagicMock()
    position2.symbol = "ETH-PERP"
    position2.quantity = Decimal("-2.0")
    position2.side = "short"
    position2.entry_price = Decimal("3000.0")
    position2.mark_price = Decimal("3200.0")

    position3 = MagicMock()
    position3.symbol = "SOL-PERP"
    position3.quantity = Decimal("0.0")  # Should be filtered out
    position3.side = "long"

    return [position1, position2, position3]


@pytest.fixture
def sample_product():
    """Sample product for testing."""
    product = MagicMock(spec=Product)
    product.symbol = "BTC-PERP"
    product.bid_price = Decimal("50000.0")
    product.ask_price = Decimal("50010.0")
    product.price = Decimal("50005.0")
    product.quote_increment = Decimal("0.1")
    return product


@pytest.fixture
def complex_balances():
    """Complex balance scenarios for edge case testing."""
    return [
        Balance(asset="USD", available=Decimal("1000.0"), total=Decimal("1000.0"), hold=Decimal("0")),
        Balance(asset="USDC", available=Decimal("0.0"), total=Decimal("0.0"), hold=Decimal("0")),
        Balance(asset="ETH", available=Decimal("-10.0"), total=Decimal("-10.0"), hold=Decimal("0")),  # Negative balance
        Balance(asset="", available=Decimal("100.0"), total=Decimal("100.0"), hold=Decimal("0")),  # Empty asset
        Balance(asset="usdc", available=Decimal("2000.0"), total=Decimal("2000.0"), hold=Decimal("0")),  # Lowercase
        Balance(asset="  BTC  ", available=Decimal("0.5"), total=Decimal("0.5"), hold=Decimal("0")),  # Whitespace
    ]


@pytest.fixture
def error_positions():
    """Positions that trigger parsing errors."""
    pos1 = MagicMock()
    pos1.symbol = "BTC-PERP"
    pos1.quantity = "invalid_quantity"  # Will cause parsing error

    pos2 = MagicMock()
    pos2.symbol = None  # Missing symbol
    pos2.quantity = Decimal("1.0")

    pos3 = MagicMock()
    pos3.symbol = "ETH-PERP"
    pos3.quantity = Decimal("0.5")
    pos3.entry_price = "not_a_decimal"  # Will cause error
    pos3.side = "long"

    return [pos1, pos2, pos3]


@pytest.fixture
def broker_with_missing_methods():
    """Broker that doesn't have optional methods."""
    broker = MagicMock(spec=IBrokerage)
    broker.list_balances.return_value = [Balance(asset="USD", available=Decimal("1000.0"), total=Decimal("1000.0"), hold=Decimal("0"))]
    broker.list_positions.return_value = []

    # Remove optional methods
    if hasattr(broker, 'get_mark_price'):
        del broker.get_mark_price
    if hasattr(broker, 'get_quote'):
        del broker.get_quote

    return broker


@pytest.fixture
def broker_with_errors():
    """Broker that raises errors in optional methods."""
    broker = MagicMock(spec=IBrokerage)
    broker.list_balances.return_value = [Balance(asset="USD", available=Decimal("1000.0"), total=Decimal("1000.0"), hold=Decimal("0"))]
    broker.list_positions.return_value = []
    broker.get_product.return_value = None  # Product not found

    # Optional methods that raise errors - create them first
    broker.get_mark_price = MagicMock(side_effect=RuntimeError("Mark price service unavailable"))
    broker.get_quote = MagicMock(side_effect=RuntimeError("Quote service unavailable"))

    return broker