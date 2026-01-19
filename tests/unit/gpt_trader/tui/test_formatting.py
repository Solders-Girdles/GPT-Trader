from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.formatting import format_price, format_quantity


class TestFormatting:
    def test_format_price(self) -> None:
        result = format_price(Decimal("44520.89963595054"), decimals=4)
        assert result == "44,520.8996"

    def test_format_quantity(self) -> None:
        result = format_quantity(Decimal("0.0001234567"), decimals=4)
        assert "0.0001" in result
