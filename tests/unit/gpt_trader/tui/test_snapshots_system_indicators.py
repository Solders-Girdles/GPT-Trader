"""
Visual regression snapshots: validation and degraded/staleness indicators.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

from gpt_trader.tui.app import TraderApp


class TestValidationDisplaySnapshots:
    """Snapshot tests for validation failure display states."""

    def test_validation_failures_warning(self, snap_compare, mock_demo_bot):
        """Snapshot test for validation failures in warning state (yellow)."""

        async def set_validation_warning(pilot):
            app = pilot.app
            app.tui_state.system_data.validation_failures = {
                "mark_staleness": 2,
                "slippage_guard": 1,
            }
            app.tui_state.system_data.validation_escalated = False
            app.tui_state.refresh()
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=set_validation_warning,
        )

    def test_validation_escalated_error(self, snap_compare, mock_demo_bot):
        """Snapshot test for validation escalated state (red)."""

        async def set_validation_escalated(pilot):
            app = pilot.app
            app.tui_state.system_data.validation_failures = {
                "mark_staleness": 5,
            }
            app.tui_state.system_data.validation_escalated = True
            app.tui_state.refresh()
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=set_validation_escalated,
        )


class TestDegradedStateSnapshots:
    """Snapshot tests for degraded and stale data states."""

    def test_degraded_mode_banner(self, snap_compare, mock_demo_bot):
        """Snapshot test for degraded mode banner."""
        import time

        async def set_degraded_mode(pilot):
            app = pilot.app
            app.tui_state.degraded_mode = True
            app.tui_state.degraded_reason = "StatusReporter unavailable"
            app.tui_state.running = True
            app.tui_state.connection_healthy = True
            app.tui_state.last_data_fetch = time.time()
            app.tui_state.refresh()
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=set_degraded_mode,
        )

    def test_connection_unhealthy_stale_data(self, snap_compare, mock_demo_bot):
        """Snapshot test for stale data with unhealthy connection."""
        import time

        async def set_connection_unhealthy(pilot):
            app = pilot.app
            app.tui_state.connection_healthy = False
            app.tui_state.running = True
            app.tui_state.degraded_mode = False
            app.tui_state.last_data_fetch = time.time() - 35
            app.tui_state.refresh()
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=set_connection_unhealthy,
        )
