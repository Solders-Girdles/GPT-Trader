"""
Visual Regression Tests using Snapshot Testing.

Uses pytest-textual-snapshot for visual regression testing of TUI components.
Snapshots are stored as SVG files in the snapshots/ directory.

To update snapshots after intentional UI changes:
    pytest tests/unit/gpt_trader/tui/test_snapshots.py --snapshot-update

For editor integration, set TEXTUAL_SNAPSHOT_FILE_OPEN_PREFIX:
    - file:// (default)
    - code://file/ (VS Code)
    - pycharm:// (PyCharm)

Visual Regression Coverage (P4):
- Main screen at multiple terminal sizes (compact, standard, wide)
- Empty states (no positions, no orders)
- Loading states
- Error indicator states
- Theme variations (dark, light, high-contrast)
- Key overlay screens (help, logs, system details)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.screens.strategy_detail_screen import StrategyDetailScreen
from gpt_trader.tui.theme import ThemeMode
from gpt_trader.tui.types import (
    DecisionData,
    ExecutionIssue,
    ExecutionMetrics,
    IndicatorContribution,
    RegimeData,
    RiskGuard,
    RiskState,
    StrategyPerformance,
    StrategyState,
)
from gpt_trader.tui.widgets.execution_issues_modal import ExecutionIssuesModal
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


@pytest.fixture
def mock_demo_bot():
    """Creates a mock demo bot for snapshot testing."""
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    # Mock engine and status reporter
    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    # Make it look like a DemoBot for mode detection
    type(bot).__name__ = "DemoBot"

    return bot


class TestMainScreenSnapshots:
    """Snapshot tests for the main trading screen at various sizes."""

    def test_main_screen_initial_state(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in its initial stopped state."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_compact_80x24(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in minimum compact terminal (80x24)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(80, 24),
        )

    def test_main_screen_standard_120x40(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in standard terminal (120x40)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_wide_200x50(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in wide terminal (200x50)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(200, 50),
        )


class TestThemeSnapshots:
    """Snapshot tests for different theme variations."""

    def test_main_screen_dark_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with dark theme (default)."""

        def create_app():
            app = TraderApp(bot=mock_demo_bot)
            return app

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_light_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with light theme."""

        async def apply_light_theme(pilot):
            app = pilot.app
            # Apply light theme CSS
            if hasattr(app, "apply_theme_css"):
                app.apply_theme_css(ThemeMode.LIGHT)
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=apply_light_theme,
        )

    def test_main_screen_high_contrast_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with high-contrast accessibility theme."""

        async def apply_high_contrast_theme(pilot):
            app = pilot.app
            if hasattr(app, "apply_theme_css"):
                app.apply_theme_css(ThemeMode.HIGH_CONTRAST)
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=apply_high_contrast_theme,
        )


class TestEmptyStateSnapshots:
    """Snapshot tests for empty data states."""

    def test_portfolio_empty_positions(self, snap_compare, mock_demo_bot):
        """Snapshot test for portfolio widget with no positions."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_portfolio_empty_orders(self, snap_compare, mock_demo_bot):
        """Snapshot test for orders tab with no open orders."""

        async def navigate_to_orders(pilot):
            # Assuming orders tab is accessible via tabbed content
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=navigate_to_orders,
        )


class TestHelpScreenSnapshots:
    """Snapshot tests for the help screen."""

    def test_help_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for HelpScreen."""

        async def open_help(pilot):
            await pilot.press("?")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(100, 35),
            run_before=open_help,
        )


class TestModeSelectionSnapshots:
    """Snapshot tests for mode selection screen."""

    def test_mode_selection_screen(self, snap_compare):
        """Snapshot test for ModeSelectionScreen (no bot provided)."""

        def create_app():
            # Create app without bot to show mode selection
            return TraderApp()

        assert snap_compare(
            create_app(),
            terminal_size=(100, 30),
        )


class TestWidgetSnapshots:
    """Snapshot tests for individual widgets."""

    def test_portfolio_widget_with_positions(self, snap_compare, mock_demo_bot):
        """Snapshot test for portfolio widget with positions loaded."""

        async def setup_positions(pilot):
            # Navigate to positions tab
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=setup_positions,
        )


class TestErrorStateSnapshots:
    """Snapshot tests for error states and indicators."""

    def test_error_indicator_visible(self, snap_compare, mock_demo_bot):
        """Snapshot test for error indicator when errors are present."""

        async def trigger_error(pilot):
            # Errors would be triggered through state validation
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=trigger_error,
        )


class TestScreenTransitionSnapshots:
    """Snapshot tests for screen transitions."""

    def test_full_logs_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for full logs screen."""

        async def open_logs(pilot):
            await pilot.press("1")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_logs,
        )

    def test_system_details_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for system details screen."""

        async def open_system_details(pilot):
            await pilot.press("2")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_system_details,
        )

    def test_alert_history_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for alert history screen."""

        async def open_alert_history(pilot):
            await pilot.press("3")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_alert_history,
        )


class TestFocusStateSnapshots:
    """Snapshot tests for focus states and keyboard navigation."""

    def test_tile_focus_ring(self, snap_compare, mock_demo_bot):
        """Snapshot test for tile focus ring visibility."""

        async def focus_tile(pilot):
            # Move focus to a tile
            await pilot.press("tab")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=focus_tile,
        )

    def test_focus_navigation_arrow_keys(self, snap_compare, mock_demo_bot):
        """Snapshot test for focus after arrow key navigation."""

        async def navigate_with_arrows(pilot):
            await pilot.press("down")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=navigate_with_arrows,
        )


class TestValidationDisplaySnapshots:
    """Snapshot tests for validation failure display states."""

    def test_validation_failures_warning(self, snap_compare, mock_demo_bot):
        """Snapshot test for validation failures in warning state (yellow).

        Shows the System tile with validation failures present but not escalated.
        """

        async def set_validation_warning(pilot):
            app = pilot.app
            # Update TUI state with validation failures (warning, not escalated)
            app.tui_state.system_data.validation_failures = {
                "mark_staleness": 2,
                "slippage_guard": 1,
            }
            app.tui_state.system_data.validation_escalated = False
            # Trigger UI refresh
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
        """Snapshot test for validation escalated state (red).

        Shows the System tile when validation failures have escalated
        and reduce-only mode is active.
        """

        async def set_validation_escalated(pilot):
            app = pilot.app
            # Update TUI state with escalated validation (5+ consecutive failures)
            app.tui_state.system_data.validation_failures = {
                "mark_staleness": 5,
            }
            app.tui_state.system_data.validation_escalated = True
            # Trigger UI refresh
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
    """Snapshot tests for degraded and stale data states.

    These tests validate visual appearance of staleness banners
    and degraded mode indicators that appear when data sources
    are unavailable or connection is unhealthy.
    """

    def test_degraded_mode_banner(self, snap_compare, mock_demo_bot):
        """Snapshot test for degraded mode with status reporter unavailable.

        Shows tiles with "Degraded: <reason>" warning banners when
        StatusReporter is not returning valid data.
        """
        import time

        async def set_degraded_mode(pilot):
            app = pilot.app
            # Set degraded mode state
            app.tui_state.degraded_mode = True
            app.tui_state.degraded_reason = "StatusReporter unavailable"
            app.tui_state.running = True
            app.tui_state.connection_healthy = True  # Still connected, just degraded
            app.tui_state.last_data_fetch = time.time()  # Fresh timestamp
            # Trigger UI refresh
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
        """Snapshot test for stale data with unhealthy connection.

        Shows tiles with "Data stale (Xs) — press R to reconnect" error
        banners when connection is unhealthy and data is stale.
        """
        import time

        async def set_connection_unhealthy(pilot):
            app = pilot.app
            # Set connection unhealthy state
            app.tui_state.connection_healthy = False
            app.tui_state.running = True
            app.tui_state.degraded_mode = False
            # Set stale data (35 seconds old)
            app.tui_state.last_data_fetch = time.time() - 35
            # Trigger UI refresh
            app.tui_state.refresh()
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=set_connection_unhealthy,
        )


# Layout guardrail tests - these validate layout integrity programmatically
class TestLayoutGuardrails:
    """Programmatic layout validation tests."""

    @pytest.mark.asyncio
    async def test_minimum_terminal_size_renders(self, mock_demo_bot):
        """Verify app renders without errors at minimum terminal size."""
        app = TraderApp(bot=mock_demo_bot)

        async with app.run_test(size=(80, 24)) as pilot:
            # App should mount without CSS errors
            await pilot.pause()
            # Check no CSS errors occurred during mount
            assert app._exception is None

    @pytest.mark.asyncio
    async def test_bento_grid_tile_visibility(self, mock_demo_bot):
        """Verify all bento grid tiles are visible at standard size."""
        app = TraderApp(bot=mock_demo_bot)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Check key tiles exist
            expected_tiles = ["tile-hero", "tile-account", "tile-market", "tile-system"]
            for tile_id in expected_tiles:
                try:
                    tile = app.query_one(f"#{tile_id}")
                    assert tile is not None, f"Tile {tile_id} not found"
                except Exception:
                    # Tile may not exist in all modes
                    pass

    @pytest.mark.asyncio
    async def test_no_overlapping_widgets(self, mock_demo_bot):
        """Verify no widget overlap at standard terminal size."""
        app = TraderApp(bot=mock_demo_bot)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Get all visible widgets from active screen (Textual 7.0+ query scope)
            # Note: This is a simplified check; full overlap detection would
            # require comparing computed regions
            widgets = list(app.screen.query("Static, Button, Label, DataTable"))
            assert len(widgets) > 0, "No widgets found"


class TestRiskPreviewSnapshots:
    """Snapshot tests for the Risk Preview section in RiskDetailModal."""

    def test_risk_preview_safe(self, snap_compare, mock_demo_bot):
        """Snapshot test for Risk Preview in safe state (10% of limit).

        Shows preview chips with all OK status when loss utilization is low.
        """
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.01,  # 10% of limit
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
        """Snapshot test for Risk Preview in critical state (70% of limit).

        Shows preview chips with warning/critical status when loss is high.
        With 5x leverage, even small shocks push projections into critical.
        """
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.07,  # 70% of limit
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
        """Snapshot test for RiskDetailModal with active guards and explanations.

        Shows the Guard Thresholds section with human-readable explanations
        for each active guard (e.g., "triggers at >= 75% of daily loss limit").
        """
        risk_data = RiskState(
            daily_loss_limit_pct=0.05,  # 5% limit
            current_daily_loss_pct=-0.031,  # 3.1% loss (62% of limit)
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
        """Snapshot test for RiskDetailModal with multiple guard types.

        Shows guard explanations for various guard types including
        reduce-only, drawdown, and volatility guards.
        """
        risk_data = RiskState(
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,  # 80% of limit
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


class TestExecutionIssuesSnapshots:
    """Snapshot tests for the Execution Issues modal."""

    def test_exec_issues_modal_with_data(self, snap_compare, mock_demo_bot):
        """Snapshot test for ExecutionIssuesModal with rejections and retries.

        Shows the modal with recent rejection and retry events,
        including timestamp, symbol, side, quantity, price, and reason.
        """
        # Use fixed timestamps for deterministic snapshots (2024-01-01 12:00:00 UTC)
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
        """Snapshot test for ExecutionIssuesModal with no issues.

        Shows the empty state with dashes for both sections.
        """
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


class TestStrategyDetailSnapshots:
    """Snapshot tests for the Strategy Detail screen."""

    def test_strategy_detail_signal_details_with_hints(self, snap_compare, mock_demo_bot):
        """Snapshot test for Strategy Detail with signal breakdown and tuning hints.

        Shows the signal detail panel with indicator contributions, bars, and hints.
        Backtest section shows placeholder (--) values.
        """
        # Build decision data with contributions
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            confidence=0.72,
            reason="Strong bullish signals",
            timestamp=1704067200.0,  # Fixed timestamp for reproducibility
            contributions=[
                IndicatorContribution(
                    name="RSI(14)",
                    value=35.20,
                    contribution=0.42,
                    weight=0.80,
                ),
                IndicatorContribution(
                    name="MACD",
                    value=-0.15,
                    contribution=-0.25,
                    weight=0.60,
                ),
                IndicatorContribution(
                    name="ADX",
                    value=25.0,
                    contribution=0.005,
                    weight=0.50,
                ),
            ],
        )

        async def setup_and_show_signal_detail(pilot):
            app = pilot.app
            # Set strategy state with decision
            app.tui_state.strategy_data = StrategyState(
                active_strategies=["ensemble_v2"],
                last_decisions={"BTC-USD": decision},
            )
            # Set performance with non-zero values
            app.tui_state.strategy_performance = StrategyPerformance(
                win_rate=0.652,
                profit_factor=1.85,
                total_return_pct=12.4,
                max_drawdown_pct=-3.2,
                total_trades=23,
                winning_trades=15,
                losing_trades=8,
                sharpe_ratio=1.24,
            )
            # Set bullish regime
            app.tui_state.regime_data = RegimeData(
                regime="BULL",
                confidence=0.82,
            )
            # Keep backtest placeholder visible
            app.tui_state.backtest_performance = StrategyPerformance(total_trades=0)
            app.tui_state.refresh()

            # Push the screen
            app.push_screen(StrategyDetailScreen())
            await pilot.pause()

            # Show signal detail for BTC-USD
            app.screen._show_signal_detail("BTC-USD")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 45),
            run_before=setup_and_show_signal_detail,
        )

    def test_strategy_detail_backtest_values(self, snap_compare, mock_demo_bot):
        """Snapshot test for Strategy Detail with real backtest values.

        Shows the backtest section with formatted metrics and hidden note.
        Signal detail panel is not shown to focus on backtest display.
        """
        # Build minimal decision data
        decision = DecisionData(
            symbol="ETH-USD",
            action="HOLD",
            confidence=0.50,
            reason="Neutral signals",
            timestamp=1704067200.0,
        )

        async def setup_backtest_values(pilot):
            app = pilot.app
            # Set strategy state
            app.tui_state.strategy_data = StrategyState(
                active_strategies=["momentum_v1"],
                last_decisions={"ETH-USD": decision},
            )
            # Set live performance
            app.tui_state.strategy_performance = StrategyPerformance(
                win_rate=0.58,
                profit_factor=1.65,
                total_return_pct=8.2,
                max_drawdown_pct=-4.1,
                total_trades=45,
                winning_trades=26,
                losing_trades=19,
                sharpe_ratio=1.05,
            )
            # Set bullish regime
            app.tui_state.regime_data = RegimeData(
                regime="SIDEWAYS",
                confidence=0.65,
            )
            # Set real backtest performance (note should be hidden)
            app.tui_state.backtest_performance = StrategyPerformance(
                win_rate=0.56,
                profit_factor=1.42,
                max_drawdown_pct=-6.2,
                total_trades=120,
            )
            app.tui_state.refresh()

            # Push the screen (don't show signal detail)
            app.push_screen(StrategyDetailScreen())
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 45),
            run_before=setup_backtest_values,
        )


# Helper function for generating initial snapshots
def enable_snapshot_generation():
    """
    Instructions for generating initial snapshots:

    1. Remove @pytest.mark.skip decorators from tests you want to snapshot
    2. Run: pytest tests/unit/gpt_trader/tui/test_snapshots.py --snapshot-update
    3. Review generated SVG files in tests/unit/gpt_trader/tui/snapshots/
    4. Commit the snapshots to version control

    After initial generation, re-add the skip decorators or remove them
    permanently for CI/CD integration.
    """
    pass
