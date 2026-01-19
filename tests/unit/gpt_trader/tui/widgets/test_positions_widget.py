from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from tests.unit.gpt_trader.tui.widgets.datatable_test_utils import (  # naming: allow
    create_mock_datatable,  # naming: allow
)
from textual.widgets import DataTable

from gpt_trader.tui.types import Position
from gpt_trader.tui.widgets.portfolio import PositionsWidget


class TestPositionsWidget:
    def test_update_positions(self) -> None:
        widget = PositionsWidget()

        mock_table = create_mock_datatable()
        mock_empty_label = MagicMock()
        widget.query_one = MagicMock(
            side_effect=lambda selector, *args: (
                mock_table
                if (selector == DataTable or "#positions" in str(selector).lower())
                else mock_empty_label
            )
        )

        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
            )
        }

        widget.update_positions(positions, Decimal("100.00"))

        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "SPOT" in str(args[1])
        assert "LONG" in str(args[2])
        assert str(args[3]) == "0.5"
        assert str(args[4]) == "50,000.0000"
        assert str(args[5]) == "50,000.0000"
        assert str(args[6]) == "$100.00"
        assert "1x" in str(args[8])
        assert kwargs.get("key") == "BTC-USD"
