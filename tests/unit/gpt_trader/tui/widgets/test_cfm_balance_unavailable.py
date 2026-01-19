"""Tests for CFMBalanceWidget when data is unavailable."""

from unittest.mock import MagicMock

from textual.widgets import Label

from gpt_trader.tui.widgets.cfm_balance import CFMBalanceWidget


class TestCFMBalanceWidgetUnavailable:
    def test_show_unavailable(self):
        """Test show_unavailable method sets has_cfm to False."""
        widget = CFMBalanceWidget()
        widget.has_cfm = True

        mock_label = MagicMock(spec=Label)
        widget.query_one = MagicMock(return_value=mock_label)

        widget.show_unavailable("Test unavailable message")

        assert widget.has_cfm is False
        mock_label.update.assert_called_once()
        call_args = mock_label.update.call_args[0][0]
        assert "Test unavailable message" in call_args
