"""Formatting helpers for the paper trading dashboard."""

from __future__ import annotations


class CurrencyFormatter:
    """Formats numeric values as US currency strings."""

    def format(self, value: float | int) -> str:
        return f"${float(value):,.2f}"


class PercentageFormatter:
    """Formats numeric values as percentage strings."""

    def format(self, value: float | int) -> str:
        percentage = float(value)
        sign = "+" if percentage > 0 else ""
        return f"{sign}{percentage:.2f}%"


class DashboardFormatter:
    """Aggregated formatter used by the dashboard."""

    def __init__(self) -> None:
        self.currency = CurrencyFormatter()
        self.percentage = PercentageFormatter()

    def currency_str(self, value: float | int) -> str:
        return self.currency.format(value)

    def percentage_str(self, value: float | int) -> str:
        return self.percentage.format(value)
