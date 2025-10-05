"""Shared fixtures for scenario-based integration tests."""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, UTC
from unittest.mock import Mock

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    Position,
    Balance,
    Quote,
)
from tests.fixtures.coinbase_factories import (
    CoinbaseOrderFactory,
    CoinbasePositionFactory,
    CoinbaseBalanceFactory,
    CoinbaseQuoteFactory,
)


@pytest.fixture
def scenario_config():
    """Standard configuration for scenario tests."""
    return BotConfig(
        profile=Profile.CANARY,
        symbols=["BTC-USD", "ETH-USD"],
        update_interval=60,
        mock_broker=True,
        dry_run=False,
        max_leverage=Decimal("3"),
        daily_loss_limit=Decimal("500.00"),
        max_trade_value=Decimal("1000.00"),
        symbol_position_caps={
            "BTC-USD": Decimal("0.05"),
            "ETH-USD": Decimal("1.0"),
        },
    )


@pytest.fixture
def funded_broker():
    """Mock broker with realistic starting capital and no positions."""
    broker = Mock()

    # $10,000 starting capital
    broker.list_balances.return_value = [
        Balance(
            asset="USD",
            total=Decimal("10000.00"),
            available=Decimal("9500.00"),
            hold=Decimal("500.00"),
        ),
    ]

    # No initial positions
    broker.list_positions.return_value = []

    # Default quotes
    broker.get_quote.side_effect = lambda symbol: Mock(
        spec=Quote,
        last=Decimal("50000.00") if "BTC" in symbol else Decimal("3000.00"),
        ts=datetime.now(UTC),
    )

    # Default successful order placement
    broker.place_order.return_value = Mock(
        order_id="default-order-123",
        status="filled",
        filled_size=Decimal("0.01"),
        average_fill_price=Decimal("50000.00"),
    )

    return broker


@pytest.fixture
def broker_with_positions(funded_broker):
    """Mock broker with existing open positions."""
    # Add existing BTC and ETH positions
    funded_broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("0.02"),
            entry_price=Decimal("48000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("40.00"),
            market_value=Decimal("1000.00"),
            side=OrderSide.BUY,
        ),
        Position(
            symbol="ETH-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("2800.00"),
            current_price=Decimal("3000.00"),
            unrealized_pnl=Decimal("100.00"),
            market_value=Decimal("1500.00"),
            side=OrderSide.BUY,
        ),
    ]

    return funded_broker


@pytest.fixture
def volatile_market_broker(funded_broker):
    """Mock broker with volatile price movements for stress testing."""
    import random
    from decimal import Decimal

    base_price_btc = Decimal("50000.00")
    base_price_eth = Decimal("3000.00")

    def volatile_quote(symbol: str) -> Mock:
        """Return quote with realistic price volatility."""
        if "BTC" in symbol:
            volatility = random.uniform(-0.05, 0.05)  # ±5% volatility
            price = base_price_btc * (Decimal("1") + Decimal(str(volatility)))
        else:
            volatility = random.uniform(-0.08, 0.08)  # ±8% volatility
            price = base_price_eth * (Decimal("1") + Decimal(str(volatility)))

        return Mock(spec=Quote, last=price, ts=datetime.now(UTC))

    funded_broker.get_quote.side_effect = volatile_quote
    return funded_broker


@pytest.fixture
def realistic_order_factory():
    """Factory for creating realistic order scenarios."""

    class OrderScenarioFactory:
        """Helper to create common order scenarios."""

        @staticmethod
        def create_successful_market_order(
            symbol: str = "BTC-USD",
            side: OrderSide = OrderSide.BUY,
            quantity: Decimal = Decimal("0.01"),
            fill_price: Decimal = Decimal("50000.00"),
        ) -> Mock:
            """Create a successfully filled market order."""
            return Mock(
                order_id=f"market-{symbol}-{side.value}",
                symbol=symbol,
                side=side,
                order_type="market",
                status="filled",
                size=quantity,
                filled_size=quantity,
                average_fill_price=fill_price,
            )

        @staticmethod
        def create_partially_filled_order(
            symbol: str = "BTC-USD",
            side: OrderSide = OrderSide.BUY,
            ordered_quantity: Decimal = Decimal("0.1"),
            filled_quantity: Decimal = Decimal("0.06"),
        ) -> Mock:
            """Create a partially filled order."""
            return Mock(
                order_id=f"partial-{symbol}",
                symbol=symbol,
                side=side,
                status="partially_filled",
                size=ordered_quantity,
                filled_size=filled_quantity,
                average_fill_price=Decimal("50000.00"),
            )

        @staticmethod
        def create_rejected_order(
            symbol: str = "BTC-USD",
            side: OrderSide = OrderSide.BUY,
            rejection_reason: str = "Insufficient funds",
        ) -> Mock:
            """Create a rejected order."""
            return Mock(
                order_id=f"rejected-{symbol}",
                symbol=symbol,
                side=side,
                status="rejected",
                rejection_reason=rejection_reason,
            )

    return OrderScenarioFactory()


@pytest.fixture
def position_scenarios():
    """Factory for creating position test scenarios."""

    class PositionScenarioFactory:
        """Helper to create common position scenarios."""

        @staticmethod
        def create_profitable_position(
            symbol: str = "BTC-USD",
            quantity: Decimal = Decimal("0.02"),
            entry_price: Decimal = Decimal("48000.00"),
            current_price: Decimal = Decimal("52000.00"),
        ) -> Position:
            """Create a position with unrealized profit."""
            unrealized_pnl = quantity * (current_price - entry_price)
            return Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                market_value=quantity * current_price,
                side=OrderSide.BUY,
            )

        @staticmethod
        def create_losing_position(
            symbol: str = "BTC-USD",
            quantity: Decimal = Decimal("0.02"),
            entry_price: Decimal = Decimal("52000.00"),
            current_price: Decimal = Decimal("48000.00"),
        ) -> Position:
            """Create a position with unrealized loss."""
            unrealized_pnl = quantity * (current_price - entry_price)
            return Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                market_value=quantity * current_price,
                side=OrderSide.BUY,
            )

        @staticmethod
        def create_multi_position_portfolio(
            symbols: list[str] = None,
        ) -> list[Position]:
            """Create a diversified portfolio with multiple positions."""
            if symbols is None:
                symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

            positions = []
            base_prices = {
                "BTC-USD": (Decimal("48000"), Decimal("50000")),
                "ETH-USD": (Decimal("2800"), Decimal("3000")),
                "SOL-USD": (Decimal("95"), Decimal("100")),
            }

            for symbol in symbols:
                if symbol in base_prices:
                    entry, current = base_prices[symbol]
                    quantity = Decimal("0.02") if "BTC" in symbol else Decimal("0.5")

                    positions.append(
                        PositionScenarioFactory.create_profitable_position(
                            symbol=symbol,
                            quantity=quantity,
                            entry_price=entry,
                            current_price=current,
                        )
                    )

            return positions

    return PositionScenarioFactory()


@pytest.fixture
def market_conditions():
    """Helper for simulating different market conditions."""

    class MarketConditions:
        """Simulate various market states for testing."""

        @staticmethod
        def create_normal_market():
            """Normal market with low volatility."""
            return {
                "volatility": Decimal("0.01"),  # 1% daily volatility
                "liquidity": "high",
                "spread": Decimal("0.0002"),  # 2 bps spread
            }

        @staticmethod
        def create_volatile_market():
            """High volatility market (stress scenario)."""
            return {
                "volatility": Decimal("0.08"),  # 8% daily volatility
                "liquidity": "medium",
                "spread": Decimal("0.002"),  # 20 bps spread
            }

        @staticmethod
        def create_illiquid_market():
            """Low liquidity market with wide spreads."""
            return {
                "volatility": Decimal("0.03"),
                "liquidity": "low",
                "spread": Decimal("0.01"),  # 100 bps spread
            }

    return MarketConditions()
