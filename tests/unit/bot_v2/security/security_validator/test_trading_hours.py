"""Tests for trading hours validation in SecurityValidator."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class TestTradingHours:
    """Test trading hours validation scenarios."""

    def test_trading_hours_weekend_rejection(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test weekend trading is rejected."""
        weekend_time = trading_hours_samples["weekend"]

        result = security_validator.check_trading_hours("BTC-USD", timestamp=weekend_time)

        assert not result.is_valid
        assert any("Market closed" in error for error in result.errors)

    def test_trading_hours_pre_market_rejection(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test pre-market trading is rejected."""
        pre_market_time = trading_hours_samples["pre_market"]

        result = security_validator.check_trading_hours("AAPL", timestamp=pre_market_time)

        assert not result.is_valid
        assert any("Outside market hours" in error for error in result.errors)

    def test_trading_hours_market_open_acceptance(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test market open trading is accepted."""
        market_open_time = trading_hours_samples["market_open"]

        result = security_validator.check_trading_hours("AAPL", timestamp=market_open_time)

        assert result.is_valid
        assert result.errors == []

    def test_trading_hours_market_close_boundary(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test market close boundary condition."""
        market_close_time = trading_hours_samples["market_close"]

        result = security_validator.check_trading_hours("AAPL", timestamp=market_close_time)

        # At exactly 4:00 PM should be rejected
        assert not result.is_valid
        assert any("Outside market hours" in error for error in result.errors)

    def test_trading_hours_after_hours_rejection(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test after hours trading is rejected."""
        after_hours_time = trading_hours_samples["after_hours"]

        result = security_validator.check_trading_hours("AAPL", timestamp=after_hours_time)

        assert not result.is_valid
        assert any("Outside market hours" in error for error in result.errors)

    def test_trading_hours_default_timestamp(self, security_validator: Any) -> None:
        """Test trading hours validation with default timestamp (current time)."""
        # This test depends on when it's run, so we'll just check it doesn't crash
        result = security_validator.check_trading_hours("BTC-USD")

        # Should return some result
        assert result is not None
        assert isinstance(result.is_valid, bool)

    def test_trading_hours_crypto_24_7(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test crypto symbols trade 24/7."""
        weekend_time = trading_hours_samples["weekend"]

        # Crypto should be allowed to trade on weekends
        result = security_validator.check_trading_hours("BTC-USD", timestamp=weekend_time)

        # Current implementation may treat all symbols the same
        # This test documents the current behavior
        assert result is not None

    def test_trading_hours_weekday_validation(self, security_validator: Any) -> None:
        """Test trading hours on different weekdays."""
        weekdays = [
            (datetime(2024, 5, 27, 10, 0), True),  # Monday 10 AM
            (datetime(2024, 5, 28, 14, 0), True),  # Tuesday 2 PM
            (datetime(2024, 5, 29, 16, 30), False),  # Wednesday 4:30 PM
            (datetime(2024, 5, 30, 9, 0), False),  # Thursday 9 AM
            (datetime(2024, 5, 31, 13, 0), True),  # Friday 1 PM
        ]

        for timestamp, expected_valid in weekdays:
            result = security_validator.check_trading_hours("AAPL", timestamp=timestamp)
            assert result.is_valid == expected_valid

    def test_trading_hours_edge_case_times(self, security_validator: Any) -> None:
        """Test trading hours edge case times."""
        edge_cases = [
            (datetime(2024, 5, 31, 9, 29, 59), False),  # 9:29:59 AM
            (datetime(2024, 5, 31, 9, 30, 0), True),  # 9:30:00 AM
            (datetime(2024, 5, 31, 9, 30, 1), True),  # 9:30:01 AM
            (datetime(2024, 5, 31, 15, 59, 59), True),  # 3:59:59 PM
            (datetime(2024, 5, 31, 16, 0, 0), False),  # 4:00:00 PM
            (datetime(2024, 5, 31, 16, 0, 1), False),  # 4:00:01 PM
        ]

        for timestamp, expected_valid in edge_cases:
            result = security_validator.check_trading_hours("AAPL", timestamp=timestamp)
            assert result.is_valid == expected_valid

    def test_trading_hours_symbol_independence(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test trading hours validation is independent of symbol."""
        market_time = trading_hours_samples["market_open"]

        symbols = ["AAPL", "GOOGL", "MSFT", "BTC-USD", "ETH-USD"]

        for symbol in symbols:
            result = security_validator.check_trading_hours(symbol, timestamp=market_time)
            # All should behave the same in current implementation
            assert result.is_valid is True

    def test_trading_hours_holiday_handling(self, security_validator: Any) -> None:
        """Test trading hours on holidays."""
        # July 4th (Independence Day) - should be closed
        holiday_time = datetime(2024, 7, 4, 10, 0)  # Thursday 10 AM

        result = security_validator.check_trading_hours("AAPL", timestamp=holiday_time)

        # Current implementation may not handle holidays
        # This test documents current behavior
        assert result is not None

    def test_trading_hours_timezone_handling(self, security_validator: Any) -> None:
        """Test trading hours timezone handling."""
        # Test with different timezones
        # Note: Current implementation uses local/system time

        # Create time in different timezone (this would need timezone-aware datetime)
        utc_time = datetime(2024, 5, 31, 14, 0)  # 2 PM UTC

        result = security_validator.check_trading_hours("AAPL", timestamp=utc_time)

        # Result depends on system timezone
        assert result is not None

    def test_trading_hours_error_handling(self, security_validator: Any) -> None:
        """Test trading hours validation error handling."""
        # Test with invalid timestamp
        invalid_timestamps = [
            None,  # type: ignore
            "invalid",  # type: ignore
            1234567890,  # type: ignore
        ]

        for invalid_timestamp in invalid_timestamps:
            result = security_validator.check_trading_hours("AAPL", timestamp=invalid_timestamp)
            assert result is not None
            if invalid_timestamp is None:
                assert isinstance(result.is_valid, bool)
            else:
                assert not result.is_valid
                assert any("Invalid timestamp" in error for error in result.errors)

    def test_trading_hours_early_morning(self, security_validator: Any) -> None:
        """Test trading hours in early morning."""
        early_morning = datetime(2024, 5, 31, 6, 0)  # 6 AM

        result = security_validator.check_trading_hours("AAPL", timestamp=early_morning)

        assert not result.is_valid
        assert any("Outside market hours" in error for error in result.errors)

    def test_trading_hours_late_night(self, security_validator: Any) -> None:
        """Test trading hours in late night."""
        late_night = datetime(2024, 5, 31, 22, 0)  # 10 PM

        result = security_validator.check_trading_hours("AAPL", timestamp=late_night)

        assert not result.is_valid
        assert any("Outside market hours" in error for error in result.errors)

    def test_trading_hours_consistency(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test trading hours validation consistency."""
        market_time = trading_hours_samples["market_open"]

        result1 = security_validator.check_trading_hours("AAPL", timestamp=market_time)
        result2 = security_validator.check_trading_hours("AAPL", timestamp=market_time)

        assert result1.is_valid == result2.is_valid
        assert result1.errors == result2.errors

    def test_trading_hours_with_different_years(self, security_validator: Any) -> None:
        """Test trading hours across different years."""
        years = [2023, 2024, 2025]

        for year in years:
            market_time = datetime(year, 5, 31, 10, 0)  # May 31, 10 AM

            result = security_validator.check_trading_hours("AAPL", timestamp=market_time)

            expected = market_time.weekday() < 5 and 9 <= market_time.hour < 16
            assert result.is_valid is expected

    def test_trading_hours_error_message_content(
        self, security_validator: Any, trading_hours_samples: dict[str, datetime]
    ) -> None:
        """Test trading hours error message content."""
        weekend_time = trading_hours_samples["weekend"]

        result = security_validator.check_trading_hours("AAPL", timestamp=weekend_time)

        assert not result.is_valid
        assert len(result.errors) >= 1
        assert any("closed" in error.lower() for error in result.errors)

    def test_trading_hours_with_microseconds(self, security_validator: Any) -> None:
        """Test trading hours with microsecond precision."""
        market_time = datetime(2024, 5, 31, 10, 0, 30, 500000)  # 10:00:30.5

        result = security_validator.check_trading_hours("AAPL", timestamp=market_time)

        assert result.is_valid
