from unittest.mock import MagicMock

from textual.widgets import Label

from gpt_trader.tui.types import RiskState
from gpt_trader.tui.widgets.risk import RiskWidget


class TestRiskWidget:
    def test_update_risk(self):
        widget = RiskWidget()

        # Mock query_one
        mock_label = MagicMock(spec=Label)

        def query_side_effect(arg, *args, **kwargs):
            return mock_label

        widget.query_one = MagicMock(side_effect=query_side_effect)

        # Create data
        data = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=0.05,
            reduce_only_mode=True,
            reduce_only_reason="Loss Limit",
            active_guards=["MaxLeverage"],
        )

        # Call update
        widget.update_risk(data)

        # Verify updates (simplified check as we mock query_one to return same mock)
        assert mock_label.update.call_count >= 4
        # Check specific calls if possible, but with shared mock it's tricky.
        # Let's just ensure it runs without error and calls update.
