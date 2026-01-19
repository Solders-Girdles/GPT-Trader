"""Tests for bar runner granularity parsing and quote conversion."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.engine.bar_runner_test_utils import (  # naming: allow
    _create_mock_candle,
    _create_mock_data_provider,
)

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner


class TestParseGranularity:
    """Tests for granularity parsing."""

    def test_one_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=1)

    def test_five_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=5)

    def test_fifteen_minute(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="FIFTEEN_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(minutes=15)

    def test_one_hour(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )
        assert runner._granularity_delta == timedelta(hours=1)

    def test_one_day(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_DAY",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        assert runner._granularity_delta == timedelta(days=1)

    def test_unsupported_granularity_raises_error(self) -> None:
        provider = _create_mock_data_provider()
        with pytest.raises(ValueError, match="Unsupported granularity"):
            ClockedBarRunner(
                data_provider=provider,
                symbols=["BTC-USD"],
                granularity="INVALID",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )


class TestBarsToQuotes:
    """Tests for bar to quote conversion."""

    def test_converts_bar_to_quote(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("50000"))
        bar_time = datetime(2024, 1, 1, 12, 0, 0)

        quotes = runner._bars_to_quotes({"BTC-USD": bar}, bar_time)

        assert "BTC-USD" in quotes
        quote = quotes["BTC-USD"]
        assert quote.symbol == "BTC-USD"
        assert quote.last == Decimal("50000")
        assert quote.ts == bar_time
        assert quote.bid < quote.last
        assert quote.ask > quote.last

    def test_spread_calculation(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bar = _create_mock_candle(symbol="BTC-USD", close=Decimal("10000"))
        quotes = runner._bars_to_quotes({"BTC-USD": bar}, datetime(2024, 1, 1))

        quote = quotes["BTC-USD"]
        spread = quote.ask - quote.bid
        expected_spread = Decimal("10000") * Decimal("10") / Decimal("20000")
        assert abs(spread - expected_spread) < Decimal("0.01")

    def test_multiple_symbols(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD", "ETH-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        bars = {
            "BTC-USD": _create_mock_candle(symbol="BTC-USD", close=Decimal("50000")),
            "ETH-USD": _create_mock_candle(symbol="ETH-USD", close=Decimal("3000")),
        }
        quotes = runner._bars_to_quotes(bars, datetime(2024, 1, 1))

        assert len(quotes) == 2
        assert "BTC-USD" in quotes
        assert "ETH-USD" in quotes
        assert quotes["BTC-USD"].last == Decimal("50000")
        assert quotes["ETH-USD"].last == Decimal("3000")
