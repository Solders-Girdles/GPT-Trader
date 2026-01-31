"""
Visual regression snapshots: execution issues and risk detail modals.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

from gpt_trader.features.live_trade.telemetry import ExecutionIssue, ExecutionMetrics
from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.execution_issues_modal import ExecutionIssuesModal
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


def _create_app(mock_demo_bot):
    return TraderApp(bot=mock_demo_bot)


class TestExecutionIssuesSnapshots:
    """Snapshot tests for the Execution Issues modal."""

    def test_exec_issues_modal_with_data(self, snap_compare, mock_demo_bot):
        """Snapshot test for ExecutionIssuesModal with rejections and retries."""
        fixed_time = 1704110400.0
        metrics = ExecutionMetrics(
            submissions_rejected=3,
            submissions_failed=2,
            retry_total=5,
            recent_rejections=[
                ExecutionIssue(
                    timestamp=fixed_time - 60,
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=0.5,
                    price=50000.0,
                    reason="rate_limit",
                ),
                ExecutionIssue(
                    timestamp=fixed_time - 120,
                    symbol="ETH-USD",
                    side="SELL",
                    quantity=2.0,
                    price=3000.0,
                    reason="insufficient_funds",
                ),
            ],
            recent_retries=[
                ExecutionIssue(
                    timestamp=fixed_time - 30,
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=0.5,
                    price=50000.0,
                    reason="timeout",
                    is_retry=True,
                ),
            ],
        )

        async def open_exec_issues_modal(pilot):
            app = pilot.app
            app.push_screen(ExecutionIssuesModal(metrics))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 35),
            run_before=open_exec_issues_modal,
        )

    def test_exec_issues_modal_empty(self, snap_compare, mock_demo_bot):
        """Snapshot test for ExecutionIssuesModal with no issues."""
        metrics = ExecutionMetrics(
            recent_rejections=[],
            recent_retries=[],
        )

        async def open_exec_issues_modal(pilot):
            app = pilot.app
            app.push_screen(ExecutionIssuesModal(metrics))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 30),
            run_before=open_exec_issues_modal,
        )


class TestRiskPreviewSnapshots:
    """Snapshot tests for the Risk Preview section in RiskDetailModal."""

    def test_risk_preview_safe(self, snap_compare, mock_demo_bot):
        """Snapshot test for Risk Preview in safe state (10% of limit)."""
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.01,
            max_leverage=1.0,
            reduce_only_mode=False,
        )

        async def open_risk_preview(pilot):
            app = pilot.app
            app.tui_state.risk_data = risk_data
            app.tui_state.refresh()
            app.push_screen(RiskDetailModal(risk_data))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 40),
            run_before=open_risk_preview,
        )

    def test_risk_preview_critical(self, snap_compare, mock_demo_bot):
        """Snapshot test for Risk Preview in critical state (70% of limit)."""
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.07,
            max_leverage=5.0,
            reduce_only_mode=True,
            reduce_only_reason="Daily loss limit",
        )

        async def open_risk_preview(pilot):
            app = pilot.app
            app.tui_state.risk_data = risk_data
            app.tui_state.refresh()
            app.push_screen(RiskDetailModal(risk_data))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 40),
            run_before=open_risk_preview,
        )


class TestGuardThresholdSnapshots:
    """Snapshot tests for the Guard Thresholds section in RiskDetailModal."""

    def test_risk_modal_with_active_guards(self, snap_compare, mock_demo_bot):
        """Snapshot test for RiskDetailModal with active guards and explanations."""
        risk_data = RiskState(
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=-0.031,
            max_leverage=5.0,
            reduce_only_mode=False,
            guards=[
                RiskGuard(name="DailyLossGuard", severity="CRITICAL"),
                RiskGuard(name="MaxLeverageGuard", severity="HIGH"),
            ],
        )

        async def open_risk_modal(pilot):
            app = pilot.app
            app.tui_state.risk_data = risk_data
            app.tui_state.refresh()
            app.push_screen(RiskDetailModal(risk_data))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 45),
            run_before=open_risk_modal,
        )

    def test_risk_modal_multiple_guard_types(self, snap_compare, mock_demo_bot):
        """Snapshot test for RiskDetailModal with multiple guard types."""
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,
            max_leverage=3.0,
            reduce_only_mode=True,
            reduce_only_reason="Daily loss limit exceeded",
            guards=[
                RiskGuard(name="DailyLossGuard", severity="CRITICAL"),
                RiskGuard(name="ReduceOnlyGuard", severity="HIGH"),
                RiskGuard(name="VolatilityGuard", severity="HIGH"),
            ],
        )

        async def open_risk_modal(pilot):
            app = pilot.app
            app.tui_state.risk_data = risk_data
            app.tui_state.refresh()
            app.push_screen(RiskDetailModal(risk_data))
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(100, 50),
            run_before=open_risk_modal,
        )
