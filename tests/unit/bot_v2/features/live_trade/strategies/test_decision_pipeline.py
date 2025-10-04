"""
Unit tests for StrategyDecisionPipeline.

Tests decision flow: guards → exits → entries → filters → sizing.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.decision_pipeline import (
    DecisionContext,
    DecisionResult,
    StrategyDecisionPipeline,
)
from bot_v2.features.live_trade.strategies.decisions import Action
from bot_v2.features.live_trade.strategies.strategy_signals import SignalSnapshot


@pytest.fixture
def mock_filters():
    """Mock market condition filters."""
    filters = Mock()
    filters.should_allow_long_entry.return_value = (True, "")
    filters.should_allow_short_entry.return_value = (True, "")
    return filters


@pytest.fixture
def mock_guards():
    """Mock risk guards."""
    guards = Mock()
    guards.check_liquidation_distance.return_value = (True, "")
    guards.check_slippage_impact.return_value = (True, "")
    return guards


@pytest.fixture
def pipeline(mock_filters, mock_guards):
    """Create decision pipeline."""
    return StrategyDecisionPipeline(
        market_filters=mock_filters, risk_guards=mock_guards, risk_manager=None
    )


@pytest.fixture
def base_context():
    """Base decision context for testing."""
    return DecisionContext(
        symbol="BTC-USD-PERP",
        signal=SignalSnapshot(
            short_ma=Decimal("105"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=True,
            bearish_cross=False,
            rsi=Decimal("50"),
        ),
        current_mark=Decimal("50000"),
        equity=Decimal("10000"),
        position_state=None,
        market_snapshot={"bid": "50000", "ask": "50010"},
        is_stale=False,
        marks=[Decimal("50000")] * 25,
        product=None,
        enable_shorts=False,
        disable_new_entries=False,
        target_leverage=2,
        trailing_stop_pct=0.01,
        long_ma_period=20,
        short_ma_period=5,
        epsilon=Decimal("1"),
        trailing_stop_state=None,
    )


class TestEarlyGuards:
    """Test early guard checks."""

    def test_stale_data_with_position_returns_reduce_only(self, pipeline, base_context):
        """Should return reduce-only hold when data is stale with position."""
        context = base_context
        context.is_stale = True
        context.position_state = {"side": "long", "quantity": "0.1"}

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert result.decision.reduce_only is True
        assert "Stale" in result.decision.reason

    def test_stale_data_without_position_rejects(self, pipeline, base_context):
        """Should reject entry when data is stale without position."""
        context = base_context
        context.is_stale = True

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert result.decision.filter_rejected is True
        assert result.rejection_type == "stale_data"

    def test_disabled_entries_blocks_new_positions(self, pipeline, base_context):
        """Should block new entries when disabled."""
        context = base_context
        context.disable_new_entries = True

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert "disabled" in result.decision.reason.lower()

    def test_insufficient_data_blocks_entry(self, pipeline, base_context):
        """Should block entry when insufficient price history."""
        context = base_context
        context.marks = [Decimal("50000")] * 10  # Less than long_ma_period

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert "Need 20 marks" in result.decision.reason


class TestPositionExits:
    """Test position exit logic."""

    def test_bearish_crossover_exits_long(self, pipeline, base_context):
        """Should exit long position on bearish crossover."""
        context = base_context
        context.position_state = {"side": "long", "quantity": "0.1"}
        context.signal = SignalSnapshot(
            short_ma=Decimal("95"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=False,
            bearish_cross=True,
            rsi=None,
        )

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.CLOSE
        assert result.decision.reduce_only is True
        assert "Bearish crossover" in result.decision.reason

    def test_bullish_crossover_exits_short(self, pipeline, base_context):
        """Should exit short position on bullish crossover."""
        context = base_context
        context.position_state = {"side": "short", "quantity": "-0.1"}

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.CLOSE
        assert result.decision.reduce_only is True
        assert "Bullish crossover" in result.decision.reason

    def test_trailing_stop_initializes_for_new_position(self, pipeline, base_context):
        """Should initialize trailing stop for new position."""
        context = base_context
        context.position_state = {"side": "long", "quantity": "0.1"}
        context.signal.bullish_cross = False  # No exit signal

        result = pipeline.evaluate(context)

        # Should initialize stop but not trigger exit
        assert result.updated_trailing_stop is not None
        assert result.updated_trailing_stop[0] == Decimal("50000")  # Peak
        assert result.updated_trailing_stop[1] == Decimal("49500")  # Stop (1% below)

    def test_trailing_stop_triggers_exit_for_long(self, pipeline, base_context):
        """Should trigger exit when trailing stop hit for long position."""
        context = base_context
        context.position_state = {"side": "long", "quantity": "0.1"}
        context.signal.bullish_cross = False
        context.trailing_stop_state = (Decimal("51000"), Decimal("50490"))  # Peak, stop
        context.current_mark = Decimal("50400")  # Below stop

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.CLOSE
        assert "Trailing stop" in result.decision.reason


class TestEntryDetermination:
    """Test entry signal logic."""

    def test_bullish_cross_generates_buy_signal(self, pipeline, base_context):
        """Should generate buy signal on bullish crossover."""
        context = base_context

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.BUY

    def test_bearish_cross_with_shorts_enabled(self, pipeline, base_context):
        """Should generate sell signal on bearish crossover when shorts enabled."""
        context = base_context
        context.enable_shorts = True
        context.signal = SignalSnapshot(
            short_ma=Decimal("95"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=False,
            bearish_cross=True,
            rsi=None,
        )

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.SELL

    def test_bearish_cross_with_shorts_disabled_holds(self, pipeline, base_context):
        """Should hold on bearish crossover when shorts disabled."""
        context = base_context
        context.signal = SignalSnapshot(
            short_ma=Decimal("95"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=False,
            bearish_cross=True,
            rsi=None,
        )

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD


class TestMarketFilters:
    """Test market filter application."""

    def test_filter_rejection_records_type(self, pipeline, base_context, mock_filters):
        """Should record rejection type from filter."""
        mock_filters.should_allow_long_entry.return_value = (False, "Spread too wide")

        result = pipeline.evaluate(base_context)

        assert result.decision.action == Action.HOLD
        assert result.decision.filter_rejected is True
        assert result.rejection_type == "filter_spread"

    def test_filter_allows_entry_when_passing(self, pipeline, base_context, mock_filters):
        """Should allow entry when filters pass."""
        mock_filters.should_allow_long_entry.return_value = (True, "")

        result = pipeline.evaluate(base_context)

        assert result.decision.action == Action.BUY


class TestRiskGuards:
    """Test risk guard checks."""

    def test_liquidation_guard_rejects_entry(self, pipeline, base_context, mock_guards):
        """Should reject entry when liquidation risk too high."""
        # Need product and market_snapshot for guards to run
        context = base_context
        context.product = Mock()
        context.product.min_size = Decimal("0.001")
        context.market_snapshot = {"bid": "50000", "ask": "50010"}

        mock_guards.check_liquidation_distance.return_value = (
            False,
            "Liquidation price too close",
        )

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert result.decision.guard_rejected is True
        assert result.rejection_type == "guard_liquidation"

    def test_slippage_guard_rejects_entry(self, pipeline, base_context, mock_guards):
        """Should reject entry when slippage impact too high."""
        # Need product and market_snapshot for guards to run
        context = base_context
        context.product = Mock()
        context.product.min_size = Decimal("0.001")
        context.market_snapshot = {"bid": "50000", "ask": "50010"}

        mock_guards.check_slippage_impact.return_value = (False, "Slippage exceeds limit")

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert result.decision.guard_rejected is True
        assert result.rejection_type == "guard_slippage"


class TestPositionSizing:
    """Test position sizing logic."""

    def test_calculates_base_position_size(self, pipeline, base_context):
        """Should calculate position size as 10% of equity with leverage."""
        result = pipeline.evaluate(base_context)

        # Equity=10000, 10% = 1000, leverage=2 => 2000
        assert result.decision.action == Action.BUY
        assert result.decision.target_notional == Decimal("2000")
        assert result.decision.leverage == 2


class TestIntegration:
    """Test end-to-end decision flows."""

    def test_full_entry_flow_with_guards_passing(self, pipeline, base_context):
        """Integration test: bullish signal → filters pass → guards pass → entry."""
        result = pipeline.evaluate(base_context)

        assert result.decision.action == Action.BUY
        assert result.decision.target_notional > 0
        assert result.decision.leverage == 2
        assert "Bullish" in result.decision.reason
        assert result.rejection_type == "entries_accepted"

    def test_full_exit_flow_trailing_stop(self, pipeline, base_context):
        """Integration test: position → trailing stop hit → exit."""
        context = base_context
        context.position_state = {"side": "long", "quantity": "0.1"}
        context.signal.bullish_cross = False
        context.trailing_stop_state = (Decimal("52000"), Decimal("51480"))
        context.current_mark = Decimal("51000")  # Below stop

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.CLOSE
        assert result.decision.reduce_only is True

    def test_hold_when_no_signal(self, pipeline, base_context):
        """Should hold when no crossover signal."""
        context = base_context
        context.signal.bullish_cross = False
        context.signal.bearish_cross = False

        result = pipeline.evaluate(context)

        assert result.decision.action == Action.HOLD
        assert "No signal" in result.decision.reason
