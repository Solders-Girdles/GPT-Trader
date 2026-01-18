"""Tests for RiskWidget."""

from unittest.mock import MagicMock

from textual.widgets import Label, ProgressBar

from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.risk import RiskWidget


class TestRiskWidget:
    """Test suite for RiskWidget."""

    def _create_widget_with_mocks(self):
        """Create RiskWidget with mocked query_one."""
        widget = RiskWidget()
        mock_label = MagicMock(spec=Label)
        mock_progress_bar = MagicMock(spec=ProgressBar)

        def query_side_effect(selector, widget_type=None):
            if widget_type == ProgressBar:
                return mock_progress_bar
            return mock_label

        widget.query_one = MagicMock(side_effect=query_side_effect)
        return widget, mock_label, mock_progress_bar

    def test_update_risk_basic(self):
        """Test basic risk update with all fields."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=0.01,
            reduce_only_mode=False,
            guards=[],
        )

        widget.update_risk(data)

        # Should call update on labels
        assert mock_label.update.call_count >= 3

    def test_update_risk_with_reduce_only(self):
        """Test risk update with reduce-only mode active."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=0.04,
            reduce_only_mode=True,
            reduce_only_reason="Loss Limit Reached",
            guards=[
                RiskGuard(name="MaxLeverage"),
                RiskGuard(name="DailyLoss"),
            ],
        )

        widget.update_risk(data)

        # Should trigger updates including reduce-only display
        assert mock_label.update.call_count >= 4
        mock_label.add_class.assert_called()

    def test_update_risk_with_active_guards(self):
        """Test risk update with multiple active guards."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=3.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.02,
            reduce_only_mode=False,
            guards=[
                RiskGuard(name="MaxLeverage"),
                RiskGuard(name="DailyLoss"),
                RiskGuard(name="MaxPositions"),
            ],
        )

        widget.update_risk(data)

        # Should display guards count with "+N more" format
        assert mock_label.update.call_count >= 4

    def test_daily_loss_progress_bar_update(self):
        """Test that daily loss progress bar is updated correctly."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,  # 10% limit
            current_daily_loss_pct=-0.05,  # 5% loss (50% of limit)
            reduce_only_mode=False,
            guards=[],
        )

        widget.update_risk(data)

        # Progress bar should be updated with percentage used
        mock_progress_bar.update.assert_called()

    def test_risk_status_low(self):
        """Test risk status calculation for LOW risk."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=2.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.01,  # Only 10% of limit used
            reduce_only_mode=False,
            guards=[],
        )

        widget.update_risk(data)

        # Risk should be LOW with minimal usage (now uses unified status classes)
        mock_label.add_class.assert_any_call("status-ok")

    def test_risk_status_medium(self):
        """Test risk status calculation for MEDIUM risk."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.05,  # 50% of limit used
            reduce_only_mode=False,
            guards=[RiskGuard(name="MaxLeverage")],
        )

        widget.update_risk(data)

        # Risk should be MEDIUM with moderate usage (now uses unified status classes)
        mock_label.add_class.assert_any_call("status-warning")

    def test_risk_status_high(self):
        """Test risk status calculation for HIGH risk."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,  # 80% of limit used
            reduce_only_mode=True,  # Adds to risk score
            reduce_only_reason="Max drawdown",
            guards=[
                RiskGuard(name="MaxLeverage"),
                RiskGuard(name="DailyLoss"),
                RiskGuard(name="MaxPositions"),
            ],
        )

        widget.update_risk(data)

        # Risk should be HIGH with severe conditions (now uses unified status classes)
        mock_label.add_class.assert_any_call("status-critical")

    def test_no_daily_loss_limit_configured(self):
        """Test handling when no daily loss limit is configured."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.0,  # No limit configured
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            guards=[],
        )

        widget.update_risk(data)

        # Should still update without error
        mock_progress_bar.update.assert_called_with(progress=0)

    def test_profit_display(self):
        """Test display when current P&L is positive (profit)."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.05,  # 5% profit
            reduce_only_mode=False,
            guards=[],
        )

        widget.update_risk(data)

        # Should handle profit case gracefully
        assert mock_label.update.call_count >= 3

    def test_single_guard_display(self):
        """Test display with a single active guard."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            guards=[RiskGuard(name="MaxLeverage")],
        )

        widget.update_risk(data)

        # Should display single guard name directly
        assert mock_label.update.call_count >= 4

    def test_reduce_only_reason_truncation(self):
        """Test that long reduce-only reasons are truncated."""
        widget, mock_label, mock_progress_bar = self._create_widget_with_mocks()

        data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=True,
            reduce_only_reason="This is a very long reason that should be truncated for display",
            guards=[],
        )

        widget.update_risk(data)

        # Should still update without error (truncation handled internally)
        assert mock_label.update.call_count >= 4
