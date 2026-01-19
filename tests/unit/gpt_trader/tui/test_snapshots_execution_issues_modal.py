from __future__ import annotations

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.types import ExecutionIssue, ExecutionMetrics
from gpt_trader.tui.widgets.execution_issues_modal import ExecutionIssuesModal


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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(100, 30),
            run_before=open_exec_issues_modal,
        )
