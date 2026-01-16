"""Integration tests for funding rate PnL in perpetuals backtesting.

These tests verify the complete funding rate pipeline:
1. SimulationConfig → FundingProcessor → SimulatedBroker → FundingPnLTracker
2. Accurate PnL calculation including funding costs
3. Long/short position funding dynamics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import (
    ConstantFundingRates,
    FundingProcessor,
)
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier, SimulationConfig
from gpt_trader.core import OrderSide, OrderType


class TestFundingPnLIntegration:
    """Integration tests for funding PnL calculation."""

    @pytest.fixture
    def simulation_config(self) -> SimulationConfig:
        """Create a simulation config with funding enabled."""
        return SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            granularity="ONE_HOUR",
            initial_equity_usd=Decimal("10000"),
            fee_tier=FeeTier.TIER_2,
            enable_funding_pnl=True,
            funding_accrual_hours=1,
            funding_settlement_hours=8,
            funding_rates_8h={"BTC-PERP-USDC": Decimal("0.0001")},  # 0.01% per 8h
        )

    @pytest.fixture
    def broker(self, simulation_config: SimulationConfig) -> SimulatedBroker:
        """Create a simulated broker."""
        broker = SimulatedBroker(
            initial_equity_usd=simulation_config.initial_equity_usd,
            fee_tier=simulation_config.fee_tier,
            config=simulation_config,
        )
        # Register a product for BTC-PERP-USDC
        from gpt_trader.core import Product

        broker.products["BTC-PERP-USDC"] = Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type="PERPETUAL",
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=20,
            expiry=None,
            funding_rate=Decimal("0.0001"),
            next_funding_time=None,
        )
        return broker

    @pytest.fixture
    def funding_processor(self, simulation_config: SimulationConfig) -> FundingProcessor:
        """Create a funding processor from config."""
        rate_provider = ConstantFundingRates(rates_8h=simulation_config.funding_rates_8h or {})
        return FundingProcessor(
            rate_provider=rate_provider,
            accrual_interval_hours=simulation_config.funding_accrual_hours,
            enabled=simulation_config.enable_funding_pnl,
        )

    def test_long_position_pays_positive_funding(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Long position pays funding when rate is positive."""
        # Setup: Set initial market price
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        # Set market data
        from gpt_trader.core import Candle, Quote

        broker._current_bar["BTC-PERP-USDC"] = Candle(
            ts=start_time,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_quote["BTC-PERP-USDC"] = Quote(
            symbol="BTC-PERP-USDC",
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            ts=start_time,
        )

        # Open a long position: Buy 0.1 BTC at $50,000
        order = broker.place_order(
            symbol="BTC-PERP-USDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        assert order.status.value in ["FILLED", "filled", "PENDING", "pending"]

        # Verify position exists
        positions = broker.list_positions()
        assert len(positions) == 1
        position = positions[0]
        assert position.symbol == "BTC-PERP-USDC"
        assert position.quantity > 0  # Long

        # Simulate 12 hours of funding at 1-hour intervals
        # Need 9+ hours for settlement to happen (first settlement at hour 1 is empty,
        # next settlement at hour 9 when 8 hours have passed since hour 1)
        for hour in range(1, 13):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time

            # Process funding
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=["BTC-PERP-USDC"],
            )

        # Check that funding was paid (reduces equity for long with positive rate)
        stats = broker.get_statistics()
        funding_pnl = stats["funding_pnl"]

        # funding_pnl tracks "funding paid" - positive means paid out (cost)
        # Long position + positive rate = pays funding = positive funding_pnl
        # After ~12 hours, we should have 1 settlement with accumulated funding
        assert funding_pnl > Decimal(
            "0"
        ), f"Expected positive funding PnL (paid), got {funding_pnl}"

    def test_short_position_receives_positive_funding(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Short position receives funding when rate is positive."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        from gpt_trader.core import Candle, Quote

        broker._current_bar["BTC-PERP-USDC"] = Candle(
            ts=start_time,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_quote["BTC-PERP-USDC"] = Quote(
            symbol="BTC-PERP-USDC",
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            ts=start_time,
        )

        # Open a short position: Sell 0.1 BTC at $50,000
        order = broker.place_order(
            symbol="BTC-PERP-USDC",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        assert order.status.value in ["FILLED", "filled", "PENDING", "pending"]

        # Verify short position
        positions = broker.list_positions()
        assert len(positions) == 1
        position = positions[0]
        assert position.quantity < 0  # Short (negative quantity)

        # Simulate 12 hours (need 9+ hours for settlement)
        for hour in range(1, 13):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=["BTC-PERP-USDC"],
            )

        stats = broker.get_statistics()
        funding_pnl = stats["funding_pnl"]

        # funding_pnl tracks "funding paid" - negative means received (profit)
        # Short position + positive rate = receives funding = negative funding_pnl
        assert funding_pnl < Decimal(
            "0"
        ), f"Expected negative funding PnL (received), got {funding_pnl}"

    def test_funding_accumulates_over_time(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Funding accumulates correctly over multiple intervals."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        from gpt_trader.core import Candle, Quote

        broker._current_bar["BTC-PERP-USDC"] = Candle(
            ts=start_time,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_quote["BTC-PERP-USDC"] = Quote(
            symbol="BTC-PERP-USDC",
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            ts=start_time,
        )

        # Open long position
        broker.place_order(
            symbol="BTC-PERP-USDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        funding_values: list[Decimal] = []

        # Process 18 hours (need 9+ for first real settlement, 17+ for second)
        for hour in range(1, 19):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=["BTC-PERP-USDC"],
            )
            funding_values.append(broker.get_statistics()["funding_pnl"])

        # Verify funding increases over time
        # Each hour should add funding (positive for long with positive rate)
        final_funding = funding_values[-1]
        assert final_funding > Decimal("0"), "Long should pay funding over time"

        # After settlement, funding_pnl should show accumulated settled funding
        # First real settlement at hour 9, second at hour 17
        assert funding_values[8] > Decimal(
            "0"
        ), "Settlement at hour 9 should show funding"  # Index 8 = hour 9
        assert final_funding > funding_values[8], "Funding should continue accumulating"

    def test_no_position_no_funding(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """No funding is charged when there's no position."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        # Don't open any position, just run funding processor
        for hour in range(1, 9):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=["BTC-PERP-USDC"],
            )

        stats = broker.get_statistics()
        assert stats["funding_pnl"] == Decimal("0")

    def test_funding_affects_final_equity(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Funding payments are reflected in final equity."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        from gpt_trader.core import Candle, Quote

        broker._current_bar["BTC-PERP-USDC"] = Candle(
            ts=start_time,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_quote["BTC-PERP-USDC"] = Quote(
            symbol="BTC-PERP-USDC",
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            ts=start_time,
        )

        _initial_equity = broker.get_equity()  # Stored for potential future assertions

        # Open long position
        broker.place_order(
            symbol="BTC-PERP-USDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        equity_after_entry = broker.get_equity()

        # Run funding for 24 hours (3 full 8-hour cycles)
        for hour in range(1, 25):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=["BTC-PERP-USDC"],
            )

        final_equity = broker.get_equity()
        stats = broker.get_statistics()
        funding_pnl = stats["funding_pnl"]

        # Final equity should reflect funding costs (for long with positive rate)
        # funding_pnl is positive (tracking amount paid), so final_equity should be less
        # than equity_after_entry by approximately the funding amount
        actual_equity_change = final_equity - equity_after_entry

        # Account for unrealized PnL (price hasn't moved, so it should be ~0)
        # The main change should be from funding (negative impact on equity)
        # funding_pnl > 0 means we paid funding (cost)
        assert funding_pnl > Decimal("0"), "Long should pay funding with positive rate"

        # Verify equity decreased (since we paid funding)
        # Note: equity_after_entry already accounts for entry fees
        # actual_equity_change should be approximately -funding_pnl
        assert actual_equity_change < Decimal("0"), "Equity should decrease from funding costs"


class TestFundingWithSimulationConfig:
    """Tests for funding configuration through SimulationConfig."""

    def test_config_funding_rates_passed_to_broker(self) -> None:
        """Verify funding rates from config are accessible."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            granularity="ONE_HOUR",
            initial_equity_usd=Decimal("10000"),
            enable_funding_pnl=True,
            funding_rates_8h={
                "BTC-PERP-USDC": Decimal("0.0001"),
                "ETH-PERP-USDC": Decimal("-0.00005"),
            },
        )

        assert config.funding_rates_8h is not None
        assert config.funding_rates_8h["BTC-PERP-USDC"] == Decimal("0.0001")
        assert config.funding_rates_8h["ETH-PERP-USDC"] == Decimal("-0.00005")

    def test_funding_disabled_skips_processing(self) -> None:
        """When enable_funding_pnl is False, no funding is processed."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            granularity="ONE_HOUR",
            initial_equity_usd=Decimal("10000"),
            enable_funding_pnl=False,
            funding_rates_8h={"BTC-PERP-USDC": Decimal("0.0001")},
        )

        processor = FundingProcessor(
            rate_provider=ConstantFundingRates(rates_8h=config.funding_rates_8h or {}),
            accrual_interval_hours=config.funding_accrual_hours,
            enabled=config.enable_funding_pnl,
        )

        assert processor.enabled is False

    def test_default_funding_settlement_is_8_hours(self) -> None:
        """Verify default settlement interval is 8 hours (Coinbase standard)."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            granularity="ONE_HOUR",
            initial_equity_usd=Decimal("10000"),
        )

        assert config.funding_settlement_hours == 8
