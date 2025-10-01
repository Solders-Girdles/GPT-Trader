"""Tests for risk_calculations.py - shared risk calculation functions.

This module tests the shared calculation utilities used across the risk system,
including daytime window evaluation, leverage caps, and maintenance margin rates.

Critical behaviors tested:
- Daytime window evaluation with timezone handling
- Effective symbol leverage cap calculation
- Effective maintenance margin rate (MMR) calculation
- Schedule-based (day/night) risk limit overrides
- Risk info provider integration
- Edge cases with invalid data and missing config
- Floating-point precision in financial calculations

Trading Safety Context:
    These calculation functions determine the actual risk limits applied
    to each trade. Incorrect calculations can result in:
    - Wrong leverage limits (allowing over-leveraged positions)
    - Incorrect margin requirements (unexpected liquidations)
    - Schedule bypasses (wrong limits applied during volatile periods)
    - Division by zero or overflow errors crashing the system

    These are pure functions that must be mathematically correct and robust.
"""

from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
    evaluate_daytime_window,
)


class TestEvaluateDaytimeWindow:
    """Test daytime window evaluation logic."""

    def test_returns_true_when_in_window(self) -> None:
        """Returns True when current time is within daytime window.

        Trading hours are configured to limit leverage during volatile periods.
        """
        # Create a config with 9:00 AM - 5:00 PM UTC window
        config = Mock()
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"

        # 12:00 PM is within window
        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now)

        assert result is True

    def test_returns_false_when_outside_window(self) -> None:
        """Returns False when current time is outside daytime window.

        Night hours may have stricter leverage limits.
        """
        config = Mock()
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"

        # 8:00 AM is before window starts
        now = datetime(2024, 1, 1, 8, 0, 0)

        result = evaluate_daytime_window(config, now)

        assert result is False

    def test_handles_window_spanning_midnight(self) -> None:
        """Handles daytime window that spans midnight.

        Example: 22:00 - 06:00 (night shift trading hours).
        """
        config = Mock()
        config.daytime_start_utc = "22:00"
        config.daytime_end_utc = "06:00"

        # 23:00 is within window (after start)
        now_night = datetime(2024, 1, 1, 23, 0, 0)
        assert evaluate_daytime_window(config, now_night) is True

        # 02:00 is within window (before end)
        now_early_morning = datetime(2024, 1, 1, 2, 0, 0)
        assert evaluate_daytime_window(config, now_early_morning) is True

        # 12:00 is outside window
        now_midday = datetime(2024, 1, 1, 12, 0, 0)
        assert evaluate_daytime_window(config, now_midday) is False

    def test_handles_exact_boundary_times(self) -> None:
        """Handles exact boundary times (inclusive start, exclusive end).

        Boundary conditions must be well-defined to avoid gaps or overlaps.
        """
        config = Mock()
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"

        # Exactly at start time (inclusive)
        at_start = datetime(2024, 1, 1, 9, 0, 0)
        assert evaluate_daytime_window(config, at_start) is True

        # Exactly at end time (exclusive)
        at_end = datetime(2024, 1, 1, 17, 0, 0)
        assert evaluate_daytime_window(config, at_end) is False

    def test_returns_none_when_no_config(self) -> None:
        """Returns None when daytime window is not configured.

        Allows system to operate without schedule-based limits.
        """
        config = Mock()
        config.daytime_start_utc = None
        config.daytime_end_utc = None

        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now)

        assert result is None

    def test_returns_none_when_start_missing(self) -> None:
        """Returns None when only start time is missing.

        Incomplete configuration should be treated as disabled.
        """
        config = Mock()
        config.daytime_start_utc = None
        config.daytime_end_utc = "17:00"

        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now)

        assert result is None

    def test_returns_none_when_end_missing(self) -> None:
        """Returns None when only end time is missing.

        Incomplete configuration should be treated as disabled.
        """
        config = Mock()
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = None

        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now)

        assert result is None

    def test_handles_invalid_time_format_gracefully(self) -> None:
        """Handles invalid time format without crashing.

        Defensive: Config may contain malformed data.
        """
        config = Mock()
        config.daytime_start_utc = "invalid"
        config.daytime_end_utc = "17:00"

        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now)

        # Should return None instead of crashing
        assert result is None

    def test_uses_current_time_when_now_is_none(self) -> None:
        """Uses current UTC time when now parameter is None.

        Convenience: Allows calling without explicit timestamp.
        """
        config = Mock()
        config.daytime_start_utc = "00:00"
        config.daytime_end_utc = "23:59"

        result = evaluate_daytime_window(config, now=None)

        # Should use datetime.utcnow() and return a boolean
        assert isinstance(result, bool)

    def test_logs_errors_when_logger_provided(self) -> None:
        """Logs debug messages when logger is provided and errors occur.

        Helps troubleshoot configuration issues in production.
        """
        config = Mock()
        config.daytime_start_utc = "invalid"
        config.daytime_end_utc = "17:00"

        logger = Mock()
        now = datetime(2024, 1, 1, 12, 0, 0)

        result = evaluate_daytime_window(config, now, logger=logger)

        assert result is None
        logger.debug.assert_called_once()


class TestEffectiveSymbolLeverageCap:
    """Test effective symbol leverage cap calculation."""

    def test_uses_default_max_leverage_when_no_overrides(self) -> None:
        """Uses config.max_leverage when symbol has no specific limit.

        Default applies to all symbols not explicitly configured.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert cap == 5

    def test_uses_symbol_specific_override(self) -> None:
        """Uses symbol-specific leverage limit when configured.

        Allows lower limits for volatile assets.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {"BTC-PERP": 3}

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert cap == 3

    def test_applies_day_leverage_override_during_day(self) -> None:
        """Applies day-specific leverage limit during daytime hours.

        Example: 5x during day, 2x at night for volatile symbols.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {}
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_leverage_max_per_symbol = {"BTC-PERP": 5}
        config.night_leverage_max_per_symbol = {}

        # Daytime (12:00 PM)
        now = datetime(2024, 1, 1, 12, 0, 0)

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=None,
        )

        assert cap == 5

    def test_applies_night_leverage_override_during_night(self) -> None:
        """Applies night-specific leverage limit during nighttime hours.

        Lower limits during less liquid hours.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {}
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_leverage_max_per_symbol = {}
        config.night_leverage_max_per_symbol = {"BTC-PERP": 2}

        # Nighttime (8:00 AM)
        now = datetime(2024, 1, 1, 8, 0, 0)

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=None,
        )

        assert cap == 2

    def test_uses_risk_info_provider_override(self) -> None:
        """Uses risk info provider's max_leverage when available.

        Exchange may provide dynamic leverage limits.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {}

        provider = Mock(return_value={"max_leverage": 3})

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert cap == 3
        provider.assert_called_once_with("BTC-PERP")

    def test_uses_risk_info_provider_leverage_cap_alias(self) -> None:
        """Uses risk info provider's 'leverage_cap' field as alias.

        Different providers may use different field names.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {}

        provider = Mock(return_value={"leverage_cap": 4})

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert cap == 4

    def test_takes_minimum_of_all_limits(self) -> None:
        """Takes minimum of all applicable limits.

        Most restrictive limit wins for safety.
        """
        config = Mock()
        config.max_leverage = 10
        config.leverage_max_per_symbol = {"BTC-PERP": 5}
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_leverage_max_per_symbol = {"BTC-PERP": 3}
        config.night_leverage_max_per_symbol = {}

        provider = Mock(return_value={"max_leverage": 2})

        # Daytime (12:00 PM)
        now = datetime(2024, 1, 1, 12, 0, 0)

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=provider,
        )

        # Minimum of: 10 (default), 5 (symbol), 3 (day), 2 (provider) = 2
        assert cap == 2

    def test_handles_provider_returning_none(self) -> None:
        """Handles provider returning None without crashing.

        Provider may not have data for all symbols.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}

        provider = Mock(return_value=None)

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert cap == 5

    def test_handles_provider_exception_gracefully(self) -> None:
        """Handles provider exception without crashing.

        Network errors or API failures should degrade gracefully.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}

        provider = Mock(side_effect=Exception("API timeout"))

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
            logger=Mock(),  # Provide logger to capture exception
        )

        # Should fall back to config values
        assert cap == 5

    def test_handles_missing_day_night_config_attributes(self) -> None:
        """Handles missing day/night config attributes gracefully.

        Older configs may not have these attributes.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        # Don't set day_leverage_max_per_symbol or night_leverage_max_per_symbol

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert cap == 5

    def test_returns_integer(self) -> None:
        """Returns integer leverage cap, not Decimal or float.

        Leverage is always expressed as whole number.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert isinstance(cap, int)


class TestEffectiveMMR:
    """Test effective maintenance margin rate (MMR) calculation."""

    def test_uses_default_mmr_when_no_overrides(self) -> None:
        """Uses config.default_maintenance_margin_rate when no overrides.

        Default MMR applies to all symbols not explicitly configured.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"  # 5%

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.05")

    def test_uses_risk_info_provider_override(self) -> None:
        """Uses risk info provider's maintenance_margin_rate when available.

        Exchange provides real-time MMR based on position size and market conditions.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        provider = Mock(return_value={"maintenance_margin_rate": "0.03"})

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert mmr == Decimal("0.03")
        provider.assert_called_once_with("BTC-PERP")

    def test_uses_risk_info_provider_mmr_alias(self) -> None:
        """Uses risk info provider's 'mmr' field as alias.

        Shorter field name for convenience.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        provider = Mock(return_value={"mmr": 0.04})

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert mmr == Decimal("0.04")

    def test_applies_day_mmr_override_during_day(self) -> None:
        """Applies day-specific MMR during daytime hours.

        May use different MMR during high-liquidity hours.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_mmr_per_symbol = {"BTC-PERP": "0.03"}
        config.night_mmr_per_symbol = {}

        # Daytime (12:00 PM)
        now = datetime(2024, 1, 1, 12, 0, 0)

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.03")

    def test_applies_night_mmr_override_during_night(self) -> None:
        """Applies night-specific MMR during nighttime hours.

        May require higher MMR during low-liquidity hours.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_mmr_per_symbol = {}
        config.night_mmr_per_symbol = {"BTC-PERP": "0.07"}

        # Nighttime (8:00 AM)
        now = datetime(2024, 1, 1, 8, 0, 0)

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.07")

    def test_provider_override_takes_precedence_over_schedule(self) -> None:
        """Risk info provider MMR takes precedence over day/night schedule.

        Real-time exchange data is most authoritative.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        config.day_mmr_per_symbol = {"BTC-PERP": "0.03"}
        config.night_mmr_per_symbol = {}

        provider = Mock(return_value={"mmr": "0.02"})

        # Daytime (12:00 PM)
        now = datetime(2024, 1, 1, 12, 0, 0)

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=now,
            risk_info_provider=provider,
        )

        # Provider MMR (0.02) takes precedence over day MMR (0.03)
        assert mmr == Decimal("0.02")

    def test_handles_provider_returning_none(self) -> None:
        """Handles provider returning None without crashing.

        Falls back to config-based MMR.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        provider = Mock(return_value=None)

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )

        assert mmr == Decimal("0.05")

    def test_handles_provider_exception_gracefully(self) -> None:
        """Handles provider exception without crashing.

        Network errors should degrade gracefully.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        provider = Mock(side_effect=Exception("API error"))

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
            logger=Mock(),
        )

        assert mmr == Decimal("0.05")

    def test_handles_missing_day_night_config_attributes(self) -> None:
        """Handles missing day/night MMR config attributes gracefully.

        Older configs may not have schedule-based MMR.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"
        config.daytime_start_utc = "09:00"
        config.daytime_end_utc = "17:00"
        # Don't set day_mmr_per_symbol or night_mmr_per_symbol

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.05")

    def test_converts_string_to_decimal(self) -> None:
        """Converts string MMR values to Decimal.

        Config often stores rates as strings.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert isinstance(mmr, Decimal)
        assert mmr == Decimal("0.05")

    def test_converts_float_to_decimal(self) -> None:
        """Converts float MMR values to Decimal.

        Provider may return floats from JSON.
        """
        config = Mock()
        config.default_maintenance_margin_rate = 0.05

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert isinstance(mmr, Decimal)
        # Float to Decimal conversion may have precision issues,
        # but should be close
        assert abs(mmr - Decimal("0.05")) < Decimal("0.0001")

    def test_returns_decimal_type(self) -> None:
        """Returns Decimal type for precision in financial calculations.

        Critical: Float arithmetic can cause rounding errors in margin calculations.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.05"

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert isinstance(mmr, Decimal)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for all calculation functions."""

    def test_functions_with_no_logger_dont_crash(self) -> None:
        """All functions work correctly with logger=None.

        Logger is optional for environments without logging.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}
        config.default_maintenance_margin_rate = "0.05"

        # Should not crash with logger=None
        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
            logger=None,
        )
        assert isinstance(cap, int)

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
            logger=None,
        )
        assert isinstance(mmr, Decimal)

    def test_provider_empty_dict_handled_gracefully(self) -> None:
        """Handles provider returning empty dict without crashing.

        Provider connected but has no data for symbol.
        """
        config = Mock()
        config.max_leverage = 5
        config.leverage_max_per_symbol = {}
        config.default_maintenance_margin_rate = "0.05"

        provider = Mock(return_value={})

        cap = effective_symbol_leverage_cap(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )
        assert cap == 5

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=provider,
        )
        assert mmr == Decimal("0.05")

    def test_very_small_mmr_values_preserved(self) -> None:
        """Very small MMR values (e.g., 0.001) are preserved precisely.

        Decimal precision prevents loss of small percentages.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.001"  # 0.1%

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.001")

    def test_very_large_mmr_values_handled(self) -> None:
        """Very large MMR values (e.g., 0.99) are handled correctly.

        Edge case: Nearly 100% margin requirement.
        """
        config = Mock()
        config.default_maintenance_margin_rate = "0.99"  # 99%

        mmr = effective_mmr(
            symbol="BTC-PERP",
            config=config,
            now=datetime(2024, 1, 1, 12, 0, 0),
            risk_info_provider=None,
        )

        assert mmr == Decimal("0.99")
