"""Tests for dashboard formatting helpers."""

from bot_v2.features.paper_trade.dashboard import DashboardFormatter


def test_currency_formatter_default() -> None:
    formatter = DashboardFormatter()
    assert formatter.currency_str(1234.56) == "$1,234.56"


def test_currency_formatter_negative() -> None:
    formatter = DashboardFormatter()
    assert formatter.currency_str(-987.65) == "$-987.65"


def test_currency_formatter_int() -> None:
    formatter = DashboardFormatter()
    assert formatter.currency_str(1000) == "$1,000.00"


def test_percentage_formatter_positive() -> None:
    formatter = DashboardFormatter()
    assert formatter.percentage_str(12.3456) == "+12.35%"


def test_percentage_formatter_negative() -> None:
    formatter = DashboardFormatter()
    assert formatter.percentage_str(-7.89) == "-7.89%"


def test_percentage_formatter_zero() -> None:
    formatter = DashboardFormatter()
    assert formatter.percentage_str(0) == "0.00%"
