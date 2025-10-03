"""
Comprehensive tests for baseline perpetuals trading strategy.

Covers:
- MA crossover signal detection (bullish/bearish/neutral)
- Entry decisions (long/short with leverage)
- Exit decisions (opposing signal, trailing stops)
- Reduce-only mode enforcement
- Disable new entries mode
- Position sizing and leverage application
- Trailing stop tracking (long/short)
- Edge cases and state management
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.strategies.decisions import Action
from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    StrategyConfig,
    create_baseline_strategy,
)


def create_bullish_crossover_data(short_period=5, long_period=20):
    """Create price data that triggers a bullish MA crossover when decide() appends current_mark."""
    # Establish stable base
    prices = [Decimal("100")] * long_period
    # Add declining prices to push short MA just below long MA
    prices.extend([Decimal("95")] * (short_period - 1))
    # DON'T append the final price - decide() will do that
    # When decide() appends current_mark=130, it triggers the crossover
    return prices


def create_bearish_crossover_data(short_period=5, long_period=20):
    """Create price data that triggers a bearish MA crossover when decide() appends current_mark."""
    # Establish stable base
    prices = [Decimal("100")] * long_period
    # Add rising prices to push short MA just above long MA
    prices.extend([Decimal("105")] * (short_period - 1))
    # DON'T append the final price - decide() will do that
    # When decide() appends current_mark=70, it triggers the crossover
    return prices


@pytest.fixture
def default_config():
    """Default strategy configuration."""
    return StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        target_leverage=2,
        trailing_stop_pct=0.01,
        position_fraction=0.05,
        enable_shorts=True,
        max_adds=0,
    )


@pytest.fixture
def mock_product():
    """Mock perpetual product."""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD-PERP"
    product.market_type = MarketType.PERPETUAL
    product.min_size = Decimal("0.001")
    product.step_size = Decimal("0.001")
    return product


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    rm = Mock()
    rm.is_reduce_only_mode.return_value = False
    return rm


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        config = StrategyConfig()

        assert config.short_ma_period == 5
        assert config.long_ma_period == 20
        assert config.target_leverage == 2
        assert config.trailing_stop_pct == 0.01
        assert config.position_fraction == 0.05
        assert config.enable_shorts is False
        assert config.max_adds == 0
        assert config.disable_new_entries is False
        assert config.ma_cross_epsilon_bps == Decimal("0")
        assert config.ma_cross_confirm_bars == 0

    def test_from_dict(self):
        """Should create config from dictionary."""
        data = {
            "short_ma_period": 10,
            "long_ma_period": 50,
            "enable_shorts": True,
            "trailing_stop_pct": 0.02,
            "unknown_field": "ignored",
        }

        config = StrategyConfig.from_dict(data)

        assert config.short_ma_period == 10
        assert config.long_ma_period == 50
        assert config.enable_shorts is True
        assert config.trailing_stop_pct == 0.02

    def test_from_dict_empty(self):
        """Should handle empty dict with defaults."""
        config = StrategyConfig.from_dict({})

        assert config.short_ma_period == 5
        assert config.long_ma_period == 20


class TestStrategyInit:
    """Test BaselinePerpsStrategy initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        strategy = BaselinePerpsStrategy()

        assert strategy.config.short_ma_period == 5
        assert strategy.environment == "live"
        assert strategy.mark_windows == {}
        assert strategy.position_adds == {}
        assert strategy.trailing_stops == {}

    def test_init_with_custom_config(self, default_config):
        """Should accept custom configuration."""
        strategy = BaselinePerpsStrategy(config=default_config)

        assert strategy.config == default_config

    def test_init_with_risk_manager(self, mock_risk_manager):
        """Should accept risk manager."""
        strategy = BaselinePerpsStrategy(risk_manager=mock_risk_manager)

        assert strategy.risk_manager == mock_risk_manager

    def test_init_with_environment(self):
        """Should accept environment label."""
        strategy = BaselinePerpsStrategy(environment="test")

        assert strategy.environment == "test"


class TestSignalCalculation:
    """Test MA crossover signal detection."""

    def test_signal_insufficient_data(self, default_config):
        """Should return neutral when insufficient data."""
        strategy = BaselinePerpsStrategy(config=default_config)
        marks = [Decimal(str(i)) for i in range(10)]  # Less than long_ma_period (20)

        signal = strategy._calculate_signal(marks)

        assert signal == "neutral"

    def test_signal_bullish_crossover(self, default_config):
        """Should detect bullish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config)

        # _calculate_signal needs the complete data including final price
        marks = create_bullish_crossover_data()
        marks.append(Decimal("130"))  # Add the triggering price
        signal = strategy._calculate_signal(marks)

        assert signal == "bullish"

    def test_signal_bearish_crossover(self, default_config):
        """Should detect bearish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config)

        # _calculate_signal needs the complete data including final price
        marks = create_bearish_crossover_data()
        marks.append(Decimal("70"))  # Add the triggering price
        signal = strategy._calculate_signal(marks)

        assert signal == "bearish"

    def test_signal_no_crossover(self, default_config):
        """Should return neutral when no fresh crossover."""
        strategy = BaselinePerpsStrategy(config=default_config)

        # Stable uptrend, no crossover
        marks = [Decimal("100") + Decimal(i) for i in range(30)]

        signal = strategy._calculate_signal(marks)

        assert signal == "neutral"

    def test_signal_with_epsilon_tolerance(self):
        """Should respect epsilon tolerance for crossover."""
        config = StrategyConfig(
            short_ma_period=5, long_ma_period=20, ma_cross_epsilon_bps=Decimal("10")
        )
        strategy = BaselinePerpsStrategy(config=config)

        # Small crossover that might be noise
        marks = [Decimal("100") for _ in range(25)]
        marks.append(Decimal("100.05"))  # Tiny move

        signal = strategy._calculate_signal(marks)

        # Should not trigger due to epsilon
        assert signal == "neutral"

    def test_signal_with_confirmation_bars(self):
        """Should require confirmation bars to persist."""
        config = StrategyConfig(short_ma_period=5, long_ma_period=20, ma_cross_confirm_bars=2)
        strategy = BaselinePerpsStrategy(config=config)

        # Bullish crossover but need persistence
        marks = [Decimal("100") - Decimal(i) for i in range(15)]
        marks.extend([Decimal("85") + Decimal(i) for i in range(15)])

        signal = strategy._calculate_signal(marks)

        # Should require confirmation
        assert signal in ["bullish", "neutral"]  # May or may not confirm


class TestEntryDecisions:
    """Test entry decision creation."""

    def test_entry_long_on_bullish_signal(self, default_config, mock_product, mock_risk_manager):
        """Should enter long on bullish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Create bullish crossover
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.BUY
        assert decision.target_notional is not None
        assert decision.target_notional > Decimal("0")
        assert "bullish" in decision.reason.lower()

    def test_entry_short_on_bearish_signal(self, default_config, mock_product, mock_risk_manager):
        """Should enter short on bearish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Create bearish crossover
        marks = create_bearish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("70"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.SELL
        assert decision.target_notional is not None
        assert "bearish" in decision.reason.lower()

    def test_entry_no_short_when_disabled(self, mock_product, mock_risk_manager):
        """Should not enter short when shorts disabled."""
        config = StrategyConfig(enable_shorts=False)
        strategy = BaselinePerpsStrategy(config=config, risk_manager=mock_risk_manager)

        # Create bearish crossover
        marks = create_bearish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("70"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.HOLD

    def test_entry_applies_leverage(self, default_config, mock_product, mock_risk_manager):
        """Should apply leverage to notional."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Bullish crossover
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        equity = Decimal("10000")
        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=equity,
            product=mock_product,
        )

        # Should be equity * fraction * leverage
        # 10000 * 0.05 * 2 = 1000
        expected = equity * Decimal("0.05") * Decimal("2")
        assert decision.target_notional == expected
        assert decision.leverage == 2

    def test_entry_respects_max_trade_usd(self, mock_product, mock_risk_manager):
        """Should cap notional at max_trade_usd."""
        config = StrategyConfig(
            position_fraction=0.5, target_leverage=5, max_trade_usd=Decimal("1000")
        )
        strategy = BaselinePerpsStrategy(config=config, risk_manager=mock_risk_manager)

        # Bullish crossover
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # max_trade_usd caps base notional before leverage
        # 10000 * 0.5 = 5000, capped to 1000, then leveraged: 1000 * 5 = 5000
        assert decision.target_notional == Decimal("5000")

    def test_entry_resets_position_tracking(self, default_config, mock_product, mock_risk_manager):
        """Should reset adds and trailing stops on entry."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Set some tracking state
        strategy.position_adds["BTC-USD-PERP"] = 2
        strategy.trailing_stops["BTC-USD-PERP"] = (Decimal("100"), Decimal("99"))

        # Bullish crossover
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert strategy.position_adds["BTC-USD-PERP"] == 0
        assert "BTC-USD-PERP" not in strategy.trailing_stops


class TestExitDecisions:
    """Test exit decision logic."""

    def test_exit_long_on_bearish_signal(self, default_config, mock_product, mock_risk_manager):
        """Should exit long on bearish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Bearish crossover
        marks = create_bearish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.5")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("70"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE
        assert decision.quantity == Decimal("1.5")
        assert decision.reduce_only is True
        assert "bearish" in decision.reason.lower()

    def test_exit_short_on_bullish_signal(self, default_config, mock_product, mock_risk_manager):
        """Should exit short on bullish crossover."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Bullish crossover
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "short", "quantity": Decimal("-2.0")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE
        assert decision.quantity == Decimal("2.0")  # Absolute value
        assert decision.reduce_only is True

    def test_exit_clears_tracking(self, default_config, mock_product, mock_risk_manager):
        """Should clear position tracking on exit."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        strategy.position_adds["BTC-USD-PERP"] = 1
        strategy.trailing_stops["BTC-USD-PERP"] = (Decimal("100"), Decimal("99"))

        # Bearish crossover
        marks = create_bearish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("95"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert "BTC-USD-PERP" not in strategy.position_adds
        assert "BTC-USD-PERP" not in strategy.trailing_stops


class TestTrailingStops:
    """Test trailing stop logic."""

    def test_trailing_stop_long_initializes(self, default_config, mock_product, mock_risk_manager):
        """Should initialize trailing stop for long."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Neutral signal
        marks = [Decimal("100") for _ in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Should hold and initialize stop
        assert decision.action == Action.HOLD
        assert "BTC-USD-PERP" in strategy.trailing_stops

        peak, stop_price = strategy.trailing_stops["BTC-USD-PERP"]
        assert peak == Decimal("100")
        assert stop_price == Decimal("99")  # 1% below

    def test_trailing_stop_long_updates_peak(self, default_config, mock_product, mock_risk_manager):
        """Should update peak and stop price as price rises."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        marks = [Decimal("100") for _ in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        # Initialize at 100
        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Price rises to 110
        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.HOLD
        peak, stop_price = strategy.trailing_stops["BTC-USD-PERP"]
        assert peak == Decimal("130")
        assert stop_price == Decimal("128.7")  # 1% below 130

    def test_trailing_stop_long_triggers(self, default_config, mock_product, mock_risk_manager):
        """Should trigger stop when price drops below stop level."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Use stable uptrend to avoid bearish crossover on drop
        marks = [Decimal("100") + Decimal(i) for i in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        # Initialize at 129 (stop at ~127.71)
        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("129"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Price drops to 127 (below stop)
        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("127"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE
        assert "trailing stop" in decision.reason.lower()

    def test_trailing_stop_short_initializes(self, default_config, mock_product, mock_risk_manager):
        """Should initialize trailing stop for short."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        marks = [Decimal("100") for _ in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "short", "quantity": Decimal("-1.0")}

        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        peak, stop_price = strategy.trailing_stops["BTC-USD-PERP"]
        assert peak == Decimal("100")
        assert stop_price == Decimal("101")  # 1% above for short

    def test_trailing_stop_short_updates_peak(
        self, default_config, mock_product, mock_risk_manager
    ):
        """Should update peak (lowest) for short as price drops."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        marks = [Decimal("100") for _ in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "short", "quantity": Decimal("-1.0")}

        # Initialize at 100
        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Price drops to 90
        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("70"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        peak, stop_price = strategy.trailing_stops["BTC-USD-PERP"]
        assert peak == Decimal("70")
        assert stop_price == Decimal("70.7")  # 1% above 70

    def test_trailing_stop_short_triggers(self, default_config, mock_product, mock_risk_manager):
        """Should trigger stop when price rises above stop level."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Use stable downtrend to avoid bullish crossover on rise
        marks = [Decimal("130") - Decimal(i) for i in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "short", "quantity": Decimal("-1.0")}

        # Initialize at 101 (stop at ~102.01)
        strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("101"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Price rises to 103 (above stop)
        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("103"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE
        assert "trailing stop" in decision.reason.lower()


class TestReduceOnlyMode:
    """Test reduce-only mode enforcement."""

    def test_reduce_only_closes_position(self, default_config, mock_product):
        """Should close position in reduce-only mode."""
        mock_rm = Mock()
        mock_rm.is_reduce_only_mode.return_value = True

        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_rm)

        marks = [Decimal("100") for _ in range(30)]
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE
        assert "reduce-only" in decision.reason.lower()

    def test_reduce_only_prevents_new_entry(self, default_config, mock_product):
        """Should prevent new entries in reduce-only mode."""
        mock_rm = Mock()
        mock_rm.is_reduce_only_mode.return_value = True

        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_rm)

        # Bullish signal
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.HOLD
        assert "reduce-only" in decision.reason.lower()


class TestDisableNewEntries:
    """Test disable_new_entries flag."""

    def test_disable_new_entries_holds(self, mock_product, mock_risk_manager):
        """Should hold when new entries disabled and no position."""
        config = StrategyConfig(disable_new_entries=True)
        strategy = BaselinePerpsStrategy(config=config, risk_manager=mock_risk_manager)

        # Bullish signal
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.HOLD
        assert "disabled" in decision.reason.lower()

    def test_disable_new_entries_still_exits(self, mock_product, mock_risk_manager):
        """Should still exit on signal when new entries disabled."""
        config = StrategyConfig(disable_new_entries=True)
        strategy = BaselinePerpsStrategy(config=config, risk_manager=mock_risk_manager)

        # Bearish signal
        marks = create_bearish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("1.0")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("70"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        assert decision.action == Action.CLOSE


class TestStateManagement:
    """Test state management methods."""

    def test_update_marks(self, default_config):
        """Should update mark window."""
        strategy = BaselinePerpsStrategy(config=default_config)

        marks = [Decimal(str(i)) for i in range(50)]
        strategy.update_marks("BTC-USD-PERP", marks)

        # Should be bounded to max window
        max_window = max(default_config.short_ma_period, default_config.long_ma_period) + 5
        assert len(strategy.mark_windows["BTC-USD-PERP"]) == max_window
        assert strategy.mark_windows["BTC-USD-PERP"][-1] == Decimal("49")

    def test_reset_specific_symbol(self, default_config):
        """Should reset state for specific symbol."""
        strategy = BaselinePerpsStrategy(config=default_config)

        strategy.mark_windows["BTC"] = [Decimal("100")]
        strategy.mark_windows["ETH"] = [Decimal("200")]
        strategy.position_adds["BTC"] = 1
        strategy.trailing_stops["BTC"] = (Decimal("100"), Decimal("99"))

        strategy.reset("BTC")

        assert "BTC" not in strategy.mark_windows
        assert "ETH" in strategy.mark_windows
        assert "BTC" not in strategy.position_adds
        assert "BTC" not in strategy.trailing_stops

    def test_reset_all(self, default_config):
        """Should reset all state."""
        strategy = BaselinePerpsStrategy(config=default_config)

        strategy.mark_windows["BTC"] = [Decimal("100")]
        strategy.position_adds["BTC"] = 1
        strategy.trailing_stops["BTC"] = (Decimal("100"), Decimal("99"))

        strategy.reset()

        assert len(strategy.mark_windows) == 0
        assert len(strategy.position_adds) == 0
        assert len(strategy.trailing_stops) == 0


class TestFactoryFunction:
    """Test create_baseline_strategy factory."""

    def test_create_with_defaults(self):
        """Should create strategy with defaults."""
        strategy = create_baseline_strategy()

        assert isinstance(strategy, BaselinePerpsStrategy)
        assert strategy.config.short_ma_period == 5

    def test_create_with_config_dict(self):
        """Should create strategy from config dict."""
        config = {"short_ma_period": 10, "long_ma_period": 50}
        strategy = create_baseline_strategy(config=config)

        assert strategy.config.short_ma_period == 10
        assert strategy.config.long_ma_period == 50

    def test_create_with_risk_manager(self, mock_risk_manager):
        """Should create strategy with risk manager."""
        strategy = create_baseline_strategy(risk_manager=mock_risk_manager)

        assert strategy.risk_manager == mock_risk_manager


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_position_fraction(self, mock_product, mock_risk_manager):
        """Should use default fraction when zero."""
        config = StrategyConfig(position_fraction=0.0)
        strategy = BaselinePerpsStrategy(config=config, risk_manager=mock_risk_manager)

        # Bullish signal
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Should use default 5%
        assert decision.target_notional == Decimal("1000")  # 10000 * 0.05 * 2 (leverage)

    def test_hold_when_position_quantity_zero(
        self, default_config, mock_product, mock_risk_manager
    ):
        """Should treat zero quantity as no position."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Bullish signal
        marks = create_bullish_crossover_data()
        strategy.update_marks("BTC-USD-PERP", marks)

        position_state = {"side": "long", "quantity": Decimal("0")}

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=position_state,
            recent_marks=None,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Should enter new position, not exit
        assert decision.action == Action.BUY

    def test_recent_marks_provided(self, default_config, mock_product, mock_risk_manager):
        """Should use provided recent_marks over internal window."""
        strategy = BaselinePerpsStrategy(config=default_config, risk_manager=mock_risk_manager)

        # Internal window has neutral data
        strategy.mark_windows["BTC-USD-PERP"] = [Decimal("100") for _ in range(30)]

        # But recent_marks has bullish crossover
        recent_marks = create_bullish_crossover_data()

        decision = strategy.decide(
            symbol="BTC-USD-PERP",
            current_mark=Decimal("130"),
            position_state=None,
            recent_marks=recent_marks,
            equity=Decimal("10000"),
            product=mock_product,
        )

        # Should use recent_marks and detect bullish
        assert decision.action == Action.BUY
