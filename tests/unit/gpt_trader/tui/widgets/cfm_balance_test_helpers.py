"""Helpers for CFMBalanceWidget tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.widgets import Label

from gpt_trader.tui.widgets.cfm_balance import CFMBalanceWidget


def make_widget_with_mock_labels() -> tuple[CFMBalanceWidget, dict[str, MagicMock]]:
    widget = CFMBalanceWidget()
    mock_labels: dict[str, MagicMock] = {
        "#cfm-balance": MagicMock(spec=Label),
        "#cfm-buying-power": MagicMock(spec=Label),
        "#cfm-avail-margin": MagicMock(spec=Label),
        "#cfm-margin-used": MagicMock(spec=Label),
        "#cfm-pnl": MagicMock(spec=Label),
        "#cfm-liq-buffer": MagicMock(spec=Label),
    }

    def query_one(selector: str, *args, **kwargs):
        return mock_labels.get(selector, MagicMock(spec=Label))

    widget.query_one = MagicMock(side_effect=query_one)
    return widget, mock_labels
