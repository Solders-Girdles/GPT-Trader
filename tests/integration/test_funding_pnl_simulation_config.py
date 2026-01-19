"""Integration tests for funding configuration through SimulationConfig."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import ConstantFundingRates, FundingProcessor
from gpt_trader.backtesting.types import SimulationConfig

pytestmark = pytest.mark.integration


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
