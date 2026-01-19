"""Tests for alert severity helpers."""

from __future__ import annotations

import pytest

from gpt_trader.monitoring.alert_types import AlertSeverity


class TestAlertSeverityNumericLevel:
    """Tests for AlertSeverity.numeric_level property."""

    def test_debug_level(self) -> None:
        assert AlertSeverity.DEBUG.numeric_level == 10

    def test_info_level(self) -> None:
        assert AlertSeverity.INFO.numeric_level == 20

    def test_warning_level(self) -> None:
        assert AlertSeverity.WARNING.numeric_level == 30

    def test_error_level(self) -> None:
        assert AlertSeverity.ERROR.numeric_level == 40

    def test_critical_level(self) -> None:
        assert AlertSeverity.CRITICAL.numeric_level == 50

    def test_levels_are_ordered(self) -> None:
        assert AlertSeverity.DEBUG.numeric_level < AlertSeverity.INFO.numeric_level
        assert AlertSeverity.INFO.numeric_level < AlertSeverity.WARNING.numeric_level
        assert AlertSeverity.WARNING.numeric_level < AlertSeverity.ERROR.numeric_level
        assert AlertSeverity.ERROR.numeric_level < AlertSeverity.CRITICAL.numeric_level


class TestAlertSeverityCoerce:
    """Tests for AlertSeverity.coerce class method."""

    def test_coerce_from_enum(self) -> None:
        result = AlertSeverity.coerce(AlertSeverity.ERROR)
        assert result is AlertSeverity.ERROR

    def test_coerce_from_numeric_10(self) -> None:
        result = AlertSeverity.coerce(10)
        assert result is AlertSeverity.DEBUG

    def test_coerce_from_numeric_20(self) -> None:
        result = AlertSeverity.coerce(20)
        assert result is AlertSeverity.INFO

    def test_coerce_from_numeric_30(self) -> None:
        result = AlertSeverity.coerce(30)
        assert result is AlertSeverity.WARNING

    def test_coerce_from_numeric_40(self) -> None:
        result = AlertSeverity.coerce(40)
        assert result is AlertSeverity.ERROR

    def test_coerce_from_numeric_50(self) -> None:
        result = AlertSeverity.coerce(50)
        assert result is AlertSeverity.CRITICAL

    def test_coerce_from_numeric_high_defaults_critical(self) -> None:
        result = AlertSeverity.coerce(100)
        assert result is AlertSeverity.CRITICAL

    def test_coerce_from_numeric_low_defaults_debug(self) -> None:
        result = AlertSeverity.coerce(5)
        assert result is AlertSeverity.DEBUG

    def test_coerce_from_string_value(self) -> None:
        result = AlertSeverity.coerce("warning")
        assert result is AlertSeverity.WARNING

    def test_coerce_from_string_name(self) -> None:
        result = AlertSeverity.coerce("WARNING")
        assert result is AlertSeverity.WARNING

    def test_coerce_strips_whitespace(self) -> None:
        result = AlertSeverity.coerce("  error  ")
        assert result is AlertSeverity.ERROR

    def test_coerce_case_insensitive(self) -> None:
        result = AlertSeverity.coerce("CrItIcAl")
        assert result is AlertSeverity.CRITICAL

    def test_coerce_raises_for_unknown_string(self) -> None:
        with pytest.raises(ValueError, match="Unknown alert severity"):
            AlertSeverity.coerce("invalid_severity")
