from __future__ import annotations

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(100, 50),
            run_before=open_risk_modal,
        )
