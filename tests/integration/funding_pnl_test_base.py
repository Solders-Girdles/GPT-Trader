"""Shared fixtures and helpers for funding PnL integration tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import ConstantFundingRates, FundingProcessor
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier, SimulationConfig

SYMBOL = "BTC-PERP-USDC"


class FundingPnLTestBase:
    """Shared setup for funding PnL integration tests."""

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
            funding_rates_8h={SYMBOL: Decimal("0.0001")},  # 0.01% per 8h
        )

    @pytest.fixture
    def broker(self, simulation_config: SimulationConfig) -> SimulatedBroker:
        """Create a simulated broker with a perpetual product registered."""
        broker = SimulatedBroker(
            initial_equity_usd=simulation_config.initial_equity_usd,
            fee_tier=simulation_config.fee_tier,
            config=simulation_config,
        )

        from gpt_trader.core import Product

        broker.products[SYMBOL] = Product(
            symbol=SYMBOL,
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

    def seed_market_data(
        self,
        broker: SimulatedBroker,
        ts: datetime,
        *,
        symbol: str = SYMBOL,
        price: Decimal = Decimal("50000"),
    ) -> None:
        """Seed broker bar/quote state for a symbol."""
        from gpt_trader.core import Candle, Quote

        broker._current_bar[symbol] = Candle(
            ts=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("100"),
        )
        broker._current_quote[symbol] = Quote(
            symbol=symbol,
            bid=price - Decimal("1"),
            ask=price + Decimal("1"),
            last=price,
            ts=ts,
        )

    def run_funding_hours(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
        start_time: datetime,
        *,
        hours: int,
        symbol: str = SYMBOL,
    ) -> None:
        """Process funding for N hours, advancing simulation time."""
        from datetime import timedelta

        for hour in range(1, hours + 1):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=[symbol],
            )

    def run_funding_hours_collect(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
        start_time: datetime,
        *,
        hours: int,
        symbol: str = SYMBOL,
    ) -> list[Decimal]:
        """Process funding for N hours, collecting broker funding_pnl after each hour."""
        from datetime import timedelta

        funding_values: list[Decimal] = []
        for hour in range(1, hours + 1):
            current_time = start_time + timedelta(hours=hour)
            broker._simulation_time = current_time
            funding_processor.process_funding(
                broker=broker,
                current_time=current_time,
                symbols=[symbol],
            )
            funding_values.append(broker.get_statistics()["funding_pnl"])

        return funding_values
