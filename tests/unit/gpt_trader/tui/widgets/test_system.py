import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Label

from gpt_trader.features.live_trade.telemetry import ExecutionMetrics
from gpt_trader.tui.types import SystemStatus
from gpt_trader.tui.widgets.system import SystemHealthWidget


class SystemHealthTestApp(App):
    def compose(self) -> ComposeResult:
        yield SystemHealthWidget(compact_mode=False)


class TestSystemHealthWidget:
    @pytest.mark.asyncio
    async def test_update_system(self) -> None:
        app = SystemHealthTestApp()

        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Initial state
            assert widget.system_data.connection_status == "UNKNOWN"

            # Update with new data
            new_data = SystemStatus(
                api_latency=123.45,
                connection_status="CONNECTED",
                rate_limit_usage="15%",
                memory_usage="42MB",
                cpu_usage="5%",
            )
            widget.update_system(new_data)

            # Verify internal state
            assert widget.system_data == new_data

            # Verify UI updates (use str() to get label content in Textual 7.0+)
            assert str(app.query_one("#connection-status", Label).render()) == "CONNECTED"
            assert app.query_one("#connection-status", Label).has_class("status-connected")
            assert str(app.query_one("#latency", Label).render()) == "123ms"
            assert (
                str(app.query_one("#rate-limit", Label).render()) == "15%"
            )  # No prefix in non-compact mode
            assert str(app.query_one("#memory", Label).render()) == "42MB"
            assert str(app.query_one("#cpu", Label).render()) == "5%"

    @pytest.mark.asyncio
    async def test_disconnected_status(self) -> None:
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)
            new_data = SystemStatus(connection_status="DISCONNECTED")
            widget.update_system(new_data)

            assert str(app.query_one("#connection-status", Label).render()) == "DISCONNECTED"
            assert app.query_one("#connection-status", Label).has_class("status-disconnected")
            assert not app.query_one("#connection-status", Label).has_class("status-connected")

    @pytest.mark.asyncio
    async def test_validation_failures_display_ok(self) -> None:
        """Test that validation failures show OK when no failures."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)
            new_data = SystemStatus(
                connection_status="CONNECTED",
                validation_failures={},
                validation_escalated=False,
            )
            widget.update_system(new_data)

            vfail_label = app.query_one("#validation-failures", Label)
            assert str(vfail_label.render()) == "OK"
            assert vfail_label.has_class("status-ok")

    @pytest.mark.asyncio
    async def test_validation_failures_display_warning(self) -> None:
        """Test that validation failures show warning style when failures exist."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)
            new_data = SystemStatus(
                connection_status="CONNECTED",
                validation_failures={"mark_staleness": 2, "slippage_guard": 1},
                validation_escalated=False,
            )
            widget.update_system(new_data)

            vfail_label = app.query_one("#validation-failures", Label)
            # Expanded mode shows "mark:2 slip:1" format
            rendered = str(vfail_label.render())
            assert "mark:2" in rendered
            assert "slip:1" in rendered
            assert vfail_label.has_class("status-warning")
            assert not vfail_label.has_class("status-error")

    @pytest.mark.asyncio
    async def test_validation_failures_display_escalated(self) -> None:
        """Test that validation failures show error style when escalated."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)
            new_data = SystemStatus(
                connection_status="CONNECTED",
                validation_failures={"mark_staleness": 5},
                validation_escalated=True,
            )
            widget.update_system(new_data)

            vfail_label = app.query_one("#validation-failures", Label)
            assert "ESCALATED" in str(vfail_label.render())
            assert vfail_label.has_class("status-error")
            assert not vfail_label.has_class("status-warning")


class TestExecutionIssues:
    """Tests for execution issues display."""

    @pytest.mark.asyncio
    async def test_execution_issues_hidden_when_empty(self) -> None:
        """Test that execution issues section is hidden when no issues."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Update with empty metrics
            metrics = ExecutionMetrics()
            widget.update_execution_metrics(metrics)

            # Section should be hidden
            issues_container = app.query_one("#execution-issues", Vertical)
            assert issues_container.has_class("hidden")

    @pytest.mark.asyncio
    async def test_execution_issues_shows_rejections(self) -> None:
        """Test that rejection reasons are displayed."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Update with rejection reasons
            metrics = ExecutionMetrics(rejection_reasons={"rate_limit": 3, "insufficient_funds": 1})
            widget.update_execution_metrics(metrics)

            # Section should be visible
            issues_container = app.query_one("#execution-issues", Vertical)
            assert not issues_container.has_class("hidden")

            # Check rejection text
            rejects_label = app.query_one("#exec-rejects", Label)
            rendered = str(rejects_label.render())
            assert "Rejects:" in rendered
            assert "rate_limit(3)" in rendered
            assert "insufficient_funds(1)" in rendered

    @pytest.mark.asyncio
    async def test_execution_issues_shows_retries(self) -> None:
        """Test that retry reasons are displayed."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Update with retry reasons
            metrics = ExecutionMetrics(retry_reasons={"timeout": 2, "network": 1})
            widget.update_execution_metrics(metrics)

            # Section should be visible
            issues_container = app.query_one("#execution-issues", Vertical)
            assert not issues_container.has_class("hidden")

            # Check retry text
            retries_label = app.query_one("#exec-retries", Label)
            rendered = str(retries_label.render())
            assert "Retries:" in rendered
            assert "timeout(2)" in rendered
            assert "network(1)" in rendered

    @pytest.mark.asyncio
    async def test_execution_issues_truncates_to_top_two(self) -> None:
        """Test that only top 2 reasons are shown with ellipsis."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Update with many rejection reasons
            metrics = ExecutionMetrics(
                rejection_reasons={
                    "rate_limit": 5,
                    "timeout": 3,
                    "network": 2,
                    "unknown": 1,
                }
            )
            widget.update_execution_metrics(metrics)

            # Check rejection text shows top 2 + ellipsis
            rejects_label = app.query_one("#exec-rejects", Label)
            rendered = str(rejects_label.render())
            assert "rate_limit(5)" in rendered
            assert "timeout(3)" in rendered
            assert "…" in rendered
            # Should not show all reasons
            assert "unknown(1)" not in rendered

    @pytest.mark.asyncio
    async def test_execution_issues_hides_empty_category(self) -> None:
        """Test that empty categories are hidden."""
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Update with only rejection reasons (no retries)
            metrics = ExecutionMetrics(rejection_reasons={"rate_limit": 1}, retry_reasons={})
            widget.update_execution_metrics(metrics)

            # Rejects should be visible
            rejects_label = app.query_one("#exec-rejects", Label)
            assert not rejects_label.has_class("hidden")

            # Retries should be hidden
            retries_label = app.query_one("#exec-retries", Label)
            assert retries_label.has_class("hidden")
