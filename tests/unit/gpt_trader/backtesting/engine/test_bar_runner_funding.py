"""Tests for bar runner funding helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.backtesting.engine.bar_runner import (
    ConstantFundingRates,
    FundingProcessor,
    IHistoricalDataProvider,
)


class TestIHistoricalDataProvider:
    """Tests for the IHistoricalDataProvider interface."""

    def test_interface_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IHistoricalDataProvider()  # type: ignore[abstract]


class TestConstantFundingRates:
    """Tests for ConstantFundingRates provider."""

    def test_returns_rate_for_known_symbol(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        rate = provider.get_rate("BTC-PERP-USDC", datetime(2024, 1, 1))
        assert rate == Decimal("0.0001")

    def test_returns_none_for_unknown_symbol(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        rate = provider.get_rate("ETH-PERP-USDC", datetime(2024, 1, 1))
        assert rate is None

    def test_ignores_time_parameter(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        rate1 = provider.get_rate("BTC-PERP-USDC", datetime(2024, 1, 1))
        rate2 = provider.get_rate("BTC-PERP-USDC", datetime(2024, 12, 31))
        assert rate1 == rate2

    def test_multiple_symbols(self) -> None:
        provider = ConstantFundingRates(
            rates_8h={
                "BTC-PERP-USDC": Decimal("0.0001"),
                "ETH-PERP-USDC": Decimal("-0.00005"),
            }
        )
        assert provider.get_rate("BTC-PERP-USDC", datetime(2024, 1, 1)) == Decimal("0.0001")
        assert provider.get_rate("ETH-PERP-USDC", datetime(2024, 1, 1)) == Decimal("-0.00005")


class TestFundingProcessor:
    """Tests for FundingProcessor."""

    def test_should_process_returns_true_for_new_symbol(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        assert processor.should_process("BTC-PERP-USDC", datetime(2024, 1, 1)) is True

    def test_should_process_returns_false_when_disabled(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, enabled=False)

        assert processor.should_process("BTC-PERP-USDC", datetime(2024, 1, 1)) is False

    def test_should_process_returns_false_before_interval(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        assert processor.should_process("BTC-PERP-USDC", datetime(2024, 1, 1, 0, 0)) is True

        processor._last_funding_time["BTC-PERP-USDC"] = datetime(2024, 1, 1, 0, 0)

        assert processor.should_process("BTC-PERP-USDC", datetime(2024, 1, 1, 0, 30)) is False

    def test_should_process_returns_true_after_interval(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        processor._last_funding_time["BTC-PERP-USDC"] = datetime(2024, 1, 1, 0, 0)

        assert processor.should_process("BTC-PERP-USDC", datetime(2024, 1, 1, 1, 0)) is True

    def test_process_funding_calls_broker(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        mock_broker = MagicMock()
        mock_broker.process_funding.return_value = Decimal("10.50")

        result = processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 0, 0),
            symbols=["BTC-PERP-USDC"],
        )

        mock_broker.process_funding.assert_called_once_with("BTC-PERP-USDC", Decimal("0.0001"))
        assert result == Decimal("10.50")

    def test_process_funding_skips_unknown_symbols(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider)

        mock_broker = MagicMock()
        mock_broker.process_funding.return_value = Decimal("0")

        result = processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 0, 0),
            symbols=["ETH-PERP-USDC"],
        )

        mock_broker.process_funding.assert_not_called()
        assert result == Decimal("0")

    def test_process_funding_tracks_total(self) -> None:
        provider = ConstantFundingRates(
            rates_8h={
                "BTC-PERP-USDC": Decimal("0.0001"),
                "ETH-PERP-USDC": Decimal("0.00005"),
            }
        )
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        mock_broker = MagicMock()
        mock_broker.process_funding.return_value = Decimal("5.00")

        processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 0, 0),
            symbols=["BTC-PERP-USDC", "ETH-PERP-USDC"],
        )

        assert processor.get_total_funding() == Decimal("10.00")

    def test_process_funding_respects_interval(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, accrual_interval_hours=1)

        mock_broker = MagicMock()
        mock_broker.process_funding.return_value = Decimal("5.00")

        processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 0, 0),
            symbols=["BTC-PERP-USDC"],
        )
        assert mock_broker.process_funding.call_count == 1

        processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 0, 30),
            symbols=["BTC-PERP-USDC"],
        )
        assert mock_broker.process_funding.call_count == 1

        processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1, 1, 0),
            symbols=["BTC-PERP-USDC"],
        )
        assert mock_broker.process_funding.call_count == 2

    def test_reset_clears_state(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider)

        processor._last_funding_time["BTC-PERP-USDC"] = datetime(2024, 1, 1)
        processor._total_funding_processed = Decimal("100")

        processor.reset()

        assert len(processor._last_funding_time) == 0
        assert processor.get_total_funding() == Decimal("0")

    def test_process_funding_disabled_returns_zero(self) -> None:
        provider = ConstantFundingRates(rates_8h={"BTC-PERP-USDC": Decimal("0.0001")})
        processor = FundingProcessor(rate_provider=provider, enabled=False)

        mock_broker = MagicMock()

        result = processor.process_funding(
            broker=mock_broker,
            current_time=datetime(2024, 1, 1),
            symbols=["BTC-PERP-USDC"],
        )

        mock_broker.process_funding.assert_not_called()
        assert result == Decimal("0")
