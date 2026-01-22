from __future__ import annotations

import time

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.screens.strategy_detail_screen import StrategyDetailScreen
from gpt_trader.tui.types import (
    DecisionData,
    IndicatorContribution,
    RegimeData,
    StrategyPerformance,
    StrategyState,
)


def _create_app(mock_demo_bot):
    return TraderApp(bot=mock_demo_bot)


class TestStrategyDetailSnapshots:
    """Snapshot tests for the Strategy Detail screen."""

    def test_strategy_detail_signal_details_with_hints(self, snap_compare, mock_demo_bot):
        """Snapshot test for Strategy Detail with signal breakdown and tuning hints."""
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            confidence=0.72,
            reason="Strong bullish signals",
            timestamp=1704067200.0,
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
            app.tui_state.strategy_data = StrategyState(
                active_strategies=["ensemble_v2"],
                last_decisions={"BTC-USD": decision},
            )
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
            app.tui_state.regime_data = RegimeData(
                regime="BULL",
                confidence=0.82,
            )
            app.tui_state.backtest_performance = StrategyPerformance(total_trades=0)
            app.tui_state.refresh()

            app.push_screen(StrategyDetailScreen())
            await pilot.pause()

            app.screen._show_signal_detail("BTC-USD")
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 45),
            run_before=setup_and_show_signal_detail,
        )

    def test_strategy_detail_backtest_values(self, snap_compare, mock_demo_bot):
        """Snapshot test for Strategy Detail with real backtest values."""
        decision = DecisionData(
            symbol="ETH-USD",
            action="HOLD",
            confidence=0.50,
            reason="Neutral signals",
            timestamp=1704067200.0,
        )

        async def setup_backtest_values(pilot):
            app = pilot.app
            app.tui_state.strategy_data = StrategyState(
                active_strategies=["momentum_v1"],
                last_decisions={"ETH-USD": decision},
            )
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
            app.tui_state.regime_data = RegimeData(
                regime="SIDEWAYS",
                confidence=0.65,
            )
            app.tui_state.backtest_performance = StrategyPerformance(
                win_rate=0.56,
                profit_factor=1.42,
                max_drawdown_pct=-6.2,
                total_trades=120,
            )
            app.tui_state.refresh()

            app.push_screen(StrategyDetailScreen())
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 45),
            run_before=setup_backtest_values,
        )


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

        assert snap_compare(
            _create_app(mock_demo_bot),
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

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=set_validation_escalated,
        )


class TestDegradedStateSnapshots:
    """Snapshot tests for degraded and stale data states."""

    def test_degraded_mode_banner(self, snap_compare, mock_demo_bot):
        """Snapshot test for degraded mode banner."""

        async def set_degraded_mode(pilot):
            app = pilot.app
            app.tui_state.degraded_mode = True
            app.tui_state.degraded_reason = "StatusReporter unavailable"
            app.tui_state.running = True
            app.tui_state.connection_healthy = True
            app.tui_state.last_data_fetch = time.time()
            app.tui_state.refresh()
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=set_degraded_mode,
        )

    def test_connection_unhealthy_stale_data(self, snap_compare, mock_demo_bot):
        """Snapshot test for stale data with unhealthy connection."""

        async def set_connection_unhealthy(pilot):
            app = pilot.app
            app.tui_state.connection_healthy = False
            app.tui_state.running = True
            app.tui_state.degraded_mode = False
            app.tui_state.last_data_fetch = time.time() - 35
            app.tui_state.refresh()
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=set_connection_unhealthy,
        )
