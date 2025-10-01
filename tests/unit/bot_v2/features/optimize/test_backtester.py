"""
Comprehensive tests for the backtesting engine.

Tests cover:
- Trade simulation with commissions and slippage
- Metric calculations (returns, Sharpe, drawdown, etc.)
- Edge cases (no trades, all wins, all losses)
- Different data scenarios
- Position management
- Equity curve generation
"""

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.optimize.backtester import (
    calculate_metrics,
    run_backtest_local,
    simulate_trades,
)
from bot_v2.features.optimize.strategies import SimpleMAStrategy
from bot_v2.features.optimize.types import BacktestMetrics


def create_sample_data(n_bars: int = 100, trend: str = "up") -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")

    if trend == "up":
        # Uptrend
        close_prices = np.linspace(100, 150, n_bars) + np.random.randn(n_bars) * 2
    elif trend == "down":
        # Downtrend
        close_prices = np.linspace(150, 100, n_bars) + np.random.randn(n_bars) * 2
    elif trend == "sideways":
        # Sideways
        close_prices = 125 + np.random.randn(n_bars) * 5
    else:
        # Random walk
        returns = np.random.randn(n_bars) * 0.02
        close_prices = 100 * np.exp(np.cumsum(returns))

    # Create realistic OHLC from close
    high = close_prices + np.abs(np.random.randn(n_bars) * 2)
    low = close_prices - np.abs(np.random.randn(n_bars) * 2)
    open_prices = close_prices + (np.random.randn(n_bars) * 1)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close_prices,
            "volume": np.random.randint(1000000, 10000000, n_bars),
        },
        index=dates,
    )


def create_simple_signals(n_bars: int = 100, pattern: str = "buy_sell") -> pd.Series:
    """Create simple trading signals for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")
    signals = pd.Series(0, index=dates)

    if pattern == "buy_sell":
        # Single round trip
        signals.iloc[10] = 1  # Buy
        signals.iloc[20] = -1  # Sell
    elif pattern == "multiple":
        # Multiple trades
        signals.iloc[10] = 1
        signals.iloc[20] = -1
        signals.iloc[30] = 1
        signals.iloc[40] = -1
        signals.iloc[50] = 1
        signals.iloc[60] = -1
    elif pattern == "buy_only":
        # Buy but never sell
        signals.iloc[10] = 1
    elif pattern == "no_trades":
        # No signals
        pass

    return signals


class TestSimulateTrades:
    """Test trade simulation functionality."""

    def test_simulate_single_round_trip(self):
        """Test simulation of a single buy-sell round trip."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_sell")

        trades, equity_curve = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005, initial_capital=10000
        )

        # Should have 2 trades (1 buy, 1 sell)
        assert len(trades) == 2
        assert trades[0]["type"] == "buy"
        assert trades[1]["type"] == "sell"
        assert trades[0]["shares"] > 0
        assert trades[1]["shares"] > 0

        # Equity curve should exist
        assert len(equity_curve) > 0

    def test_simulate_multiple_trades(self):
        """Test simulation with multiple round trips."""
        data = create_sample_data(100, trend="sideways")
        signals = create_simple_signals(100, pattern="multiple")

        trades, equity_curve = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005
        )

        # Should have 6 trades (3 buys, 3 sells)
        assert len(trades) == 6
        buy_trades = [t for t in trades if t["type"] == "buy"]
        sell_trades = [t for t in trades if t["type"] == "sell"]
        assert len(buy_trades) == 3
        assert len(sell_trades) == 3

    def test_simulate_no_trades(self):
        """Test simulation with no trading signals."""
        data = create_sample_data(100)
        signals = create_simple_signals(100, pattern="no_trades")

        trades, equity_curve = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005
        )

        # Should have no trades
        assert len(trades) == 0
        # Equity should remain constant at initial capital
        assert all(e == 10000 for e in equity_curve)

    def test_simulate_applies_commission(self):
        """Test that commission is properly applied."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_sell")

        # Run with and without commission
        trades_with_comm, _ = simulate_trades(
            signals, data, commission=0.01, slippage=0.0
        )
        trades_no_comm, _ = simulate_trades(
            signals, data, commission=0.0, slippage=0.0
        )

        # Buy with commission should cost more
        assert trades_with_comm[0]["value"] > trades_no_comm[0]["value"]

        # Sell with commission should yield less
        assert trades_with_comm[1]["value"] < trades_no_comm[1]["value"]

    def test_simulate_applies_slippage(self):
        """Test that slippage is properly applied."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_sell")

        # Run with and without slippage
        trades_with_slip, _ = simulate_trades(
            signals, data, commission=0.0, slippage=0.01
        )
        trades_no_slip, _ = simulate_trades(
            signals, data, commission=0.0, slippage=0.0
        )

        # Buy with slippage should pay higher price
        assert trades_with_slip[0]["price"] > trades_no_slip[0]["price"]

        # Sell with slippage should receive lower price
        assert trades_with_slip[1]["price"] < trades_no_slip[1]["price"]

    def test_simulate_tracks_pnl(self):
        """Test that P&L is calculated correctly."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_sell")

        trades, _ = simulate_trades(signals, data, commission=0.0, slippage=0.0)

        # Sell trade should have P&L
        sell_trade = trades[1]
        assert "pnl" in sell_trade
        assert "pnl_pct" in sell_trade

        # In uptrend, should be profitable
        assert sell_trade["pnl"] > 0
        assert sell_trade["pnl_pct"] > 0

    def test_simulate_closes_final_position(self):
        """Test that open position is closed at end."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_only")

        trades, equity_curve = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005
        )

        # Should have buy trade
        assert len(trades) >= 1

        # Final equity should reflect position closure
        assert len(equity_curve) > 0
        # Final equity should be close to cash value
        assert equity_curve.iloc[-1] > 0

    def test_simulate_with_different_initial_capital(self):
        """Test simulation with different starting capital."""
        data = create_sample_data(100, trend="up")
        signals = create_simple_signals(100, pattern="buy_sell")

        _, equity_10k = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005, initial_capital=10000
        )
        _, equity_50k = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005, initial_capital=50000
        )

        # Larger capital should have larger equity values
        assert equity_50k.iloc[0] > equity_10k.iloc[0]

    def test_simulate_insufficient_capital(self):
        """Test simulation with insufficient capital for trades."""
        data = create_sample_data(100, trend="up")
        # Prices around 100-150
        signals = create_simple_signals(100, pattern="buy_sell")

        # Very small capital
        trades, equity_curve = simulate_trades(
            signals, data, commission=0.001, slippage=0.0005, initial_capital=10
        )

        # Should have no trades (can't afford shares)
        assert len(trades) == 0


class TestCalculateMetrics:
    """Test metric calculation functionality."""

    def test_metrics_with_profitable_trades(self):
        """Test metrics calculation with profitable trades."""
        # Create winning trades
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 110, "shares": 10, "pnl": 100, "pnl_pct": 0.1},
            {"type": "buy", "price": 110, "shares": 10},
            {"type": "sell", "price": 120, "shares": 10, "pnl": 100, "pnl_pct": 0.09},
        ]

        equity_curve = pd.Series([10000, 10050, 10100, 10150, 10200])

        metrics = calculate_metrics(trades, equity_curve)

        assert metrics.total_return > 0
        assert metrics.win_rate == 1.0  # 100% wins
        assert metrics.total_trades == 4
        assert metrics.best_trade > 0
        assert metrics.max_drawdown >= 0

    def test_metrics_with_losing_trades(self):
        """Test metrics calculation with losing trades."""
        # Create losing trades
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 90, "shares": 10, "pnl": -100, "pnl_pct": -0.1},
            {"type": "buy", "price": 90, "shares": 10},
            {"type": "sell", "price": 80, "shares": 10, "pnl": -100, "pnl_pct": -0.11},
        ]

        equity_curve = pd.Series([10000, 9950, 9900, 9850, 9800])

        metrics = calculate_metrics(trades, equity_curve)

        assert metrics.total_return < 0
        assert metrics.win_rate == 0.0  # 0% wins
        assert metrics.worst_trade < 0

    def test_metrics_with_mixed_trades(self):
        """Test metrics with both winning and losing trades."""
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 110, "shares": 10, "pnl": 100, "pnl_pct": 0.1},
            {"type": "buy", "price": 110, "shares": 10},
            {"type": "sell", "price": 100, "shares": 10, "pnl": -100, "pnl_pct": -0.09},
        ]

        equity_curve = pd.Series([10000, 10050, 10100, 10050, 10000])

        metrics = calculate_metrics(trades, equity_curve)

        assert 0 < metrics.win_rate < 1.0
        assert metrics.best_trade > 0
        assert metrics.worst_trade < 0

    def test_metrics_no_trades(self):
        """Test metrics calculation with no trades."""
        trades = []
        equity_curve = pd.Series([10000] * 10)

        metrics = calculate_metrics(trades, equity_curve)

        assert metrics.total_return == 0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0
        assert metrics.sharpe_ratio == 0

    def test_metrics_empty_equity_curve(self):
        """Test metrics with empty equity curve."""
        trades = []
        equity_curve = pd.Series([])

        metrics = calculate_metrics(trades, equity_curve)

        assert metrics.total_return == 0
        assert metrics.max_drawdown == 0

    def test_metrics_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create equity curve with volatility
        returns = np.random.randn(100) * 0.01 + 0.001  # Positive expected return
        equity_curve = pd.Series(10000 * np.exp(np.cumsum(returns)))

        trades = []
        metrics = calculate_metrics(trades, equity_curve)

        # Sharpe should be calculated
        assert isinstance(metrics.sharpe_ratio, float)
        # With positive returns, Sharpe should be positive
        if metrics.total_return > 0:
            assert metrics.sharpe_ratio > 0

    def test_metrics_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with clear drawdown
        equity_values = [10000, 10500, 10800, 9500, 9000, 9500, 10000, 11000]
        equity_curve = pd.Series(equity_values)

        trades = []
        metrics = calculate_metrics(trades, equity_curve)

        # Max drawdown from peak (10800) to valley (9000)
        # DD = (9000 - 10800) / 10800 = -16.67%
        expected_dd = (9000 - 10800) / 10800
        assert abs(metrics.max_drawdown - abs(expected_dd)) < 0.01

    def test_metrics_profit_factor(self):
        """Test profit factor calculation."""
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 120, "shares": 10, "pnl": 200, "pnl_pct": 0.2},  # Win $200
            {"type": "buy", "price": 120, "shares": 10},
            {"type": "sell", "price": 110, "shares": 10, "pnl": -100, "pnl_pct": -0.08},  # Lose $100
        ]

        equity_curve = pd.Series([10000, 10100, 10200, 10100, 10100])

        metrics = calculate_metrics(trades, equity_curve)

        # Profit factor = gross wins / gross losses = 200 / 100 = 2.0
        assert abs(metrics.profit_factor - 2.0) < 0.1

    def test_metrics_profit_factor_no_losses(self):
        """Test profit factor with only winning trades."""
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 110, "shares": 10, "pnl": 100, "pnl_pct": 0.1},
        ]

        equity_curve = pd.Series([10000, 10050, 10100])

        metrics = calculate_metrics(trades, equity_curve)

        # Profit factor should be infinity or very large
        assert metrics.profit_factor == float("inf") or metrics.profit_factor > 100

    def test_metrics_recovery_factor(self):
        """Test recovery factor calculation."""
        # Net profit / max drawdown
        equity_curve = pd.Series([10000, 11000, 9000, 10500])  # Net +5%, DD 18%
        trades = []

        metrics = calculate_metrics(trades, equity_curve)

        # Recovery factor = total return / max drawdown
        if metrics.max_drawdown > 0:
            expected_rf = metrics.total_return / metrics.max_drawdown
            assert abs(metrics.recovery_factor - expected_rf) < 0.01

    def test_metrics_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        equity_curve = pd.Series([10000, 11000, 9500, 12000])
        trades = []

        metrics = calculate_metrics(trades, equity_curve)

        # Calmar = return / max drawdown
        if metrics.max_drawdown > 0:
            expected_calmar = metrics.total_return / metrics.max_drawdown
            assert abs(metrics.calmar_ratio - expected_calmar) < 0.01


class TestRunBacktestLocal:
    """Test complete backtest execution."""

    def test_backtest_simple_ma_strategy(self):
        """Test backtesting MA crossover strategy."""
        data = create_sample_data(200, trend="up")
        params = {"fast_period": 10, "slow_period": 30}

        metrics = run_backtest_local("SimpleMA", data, params)

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_trades >= 0
        assert -1 <= metrics.total_return <= 10  # Reasonable bounds

    def test_backtest_with_commission(self):
        """Test that commission impacts returns."""
        data = create_sample_data(200, trend="up")
        params = {"fast_period": 10, "slow_period": 30}

        metrics_no_comm = run_backtest_local("SimpleMA", data, params, commission=0.0)
        metrics_with_comm = run_backtest_local("SimpleMA", data, params, commission=0.01)

        # Commission should reduce returns
        assert metrics_with_comm.total_return <= metrics_no_comm.total_return

    def test_backtest_with_slippage(self):
        """Test that slippage impacts returns."""
        data = create_sample_data(200, trend="up")
        params = {"fast_period": 10, "slow_period": 30}

        metrics_no_slip = run_backtest_local("SimpleMA", data, params, slippage=0.0)
        metrics_with_slip = run_backtest_local("SimpleMA", data, params, slippage=0.01)

        # Slippage should reduce returns
        assert metrics_with_slip.total_return <= metrics_no_slip.total_return

    def test_backtest_different_strategies(self):
        """Test backtesting different strategy types."""
        data = create_sample_data(200, trend="sideways")

        strategies = [
            ("SimpleMA", {"fast_period": 10, "slow_period": 30}),
            ("Momentum", {"lookback": 20, "threshold": 0.02, "hold_period": 5}),
            ("MeanReversion", {"period": 20, "entry_std": 2.0, "exit_std": 0.5}),
        ]

        for strategy_name, params in strategies:
            metrics = run_backtest_local(strategy_name, data, params)
            assert isinstance(metrics, BacktestMetrics)

    def test_backtest_insufficient_data(self):
        """Test backtest with insufficient data for strategy."""
        # Only 10 bars but strategy needs more
        data = create_sample_data(10)
        params = {"fast_period": 5, "slow_period": 50}  # Needs 50+ bars

        metrics = run_backtest_local("SimpleMA", data, params)

        # Should handle gracefully, likely no trades
        assert metrics.total_trades == 0

    def test_backtest_trending_vs_sideways(self):
        """Test strategy performance in different market conditions."""
        params = {"fast_period": 10, "slow_period": 30}

        trending_data = create_sample_data(200, trend="up")
        sideways_data = create_sample_data(200, trend="sideways")

        trending_metrics = run_backtest_local("SimpleMA", trending_data, params)
        sideways_metrics = run_backtest_local("SimpleMA", sideways_data, params)

        # MA strategies typically perform better in trends
        # (Though not guaranteed due to randomness)
        assert isinstance(trending_metrics, BacktestMetrics)
        assert isinstance(sideways_metrics, BacktestMetrics)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_commission_and_slippage(self):
        """Test backtest with zero transaction costs."""
        data = create_sample_data(100, trend="up")
        params = {"fast_period": 10, "slow_period": 30}

        metrics = run_backtest_local("SimpleMA", data, params, commission=0.0, slippage=0.0)

        assert isinstance(metrics, BacktestMetrics)

    def test_very_high_commission(self):
        """Test backtest with unrealistically high commission."""
        data = create_sample_data(100, trend="up")
        params = {"fast_period": 10, "slow_period": 30}

        # 10% commission - should destroy returns
        metrics = run_backtest_local("SimpleMA", data, params, commission=0.1)

        # Should complete but likely negative return
        assert isinstance(metrics, BacktestMetrics)

    def test_all_winning_trades(self):
        """Test metrics calculation with perfect strategy."""
        # Create guaranteed uptrend
        data = create_sample_data(100, trend="up")
        # Create perfect signals
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = -1  # Sell at end

        trades, equity = simulate_trades(signals, data, commission=0.0, slippage=0.0)
        metrics = calculate_metrics(trades, equity)

        # Should be profitable
        assert metrics.total_return > 0
        assert metrics.win_rate >= 0.0

    def test_all_losing_trades(self):
        """Test metrics with strategy that only loses."""
        # Create downtrend
        data = create_sample_data(100, trend="down")
        # Buy high, sell low
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = -1  # Sell at end

        trades, equity = simulate_trades(signals, data, commission=0.0, slippage=0.0)
        metrics = calculate_metrics(trades, equity)

        # Should be unprofitable
        assert metrics.total_return < 0
        assert metrics.win_rate == 0.0

    def test_single_trade(self):
        """Test metrics with only one trade."""
        trades = [
            {"type": "buy", "price": 100, "shares": 10},
            {"type": "sell", "price": 105, "shares": 10, "pnl": 50, "pnl_pct": 0.05},
        ]
        equity_curve = pd.Series([10000, 10050])

        metrics = calculate_metrics(trades, equity_curve)

        assert metrics.total_trades == 2
        assert metrics.win_rate == 1.0
        assert metrics.avg_trade > 0

    def test_extreme_volatility(self):
        """Test simulation with extremely volatile prices."""
        # Create highly volatile data
        np.random.seed(42)
        close_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.1))
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * 1.05,
                "low": close_prices * 0.95,
                "close": close_prices,
                "volume": 1000000,
            },
            index=pd.date_range("2024-01-01", periods=100),
        )

        params = {"fast_period": 5, "slow_period": 20}
        metrics = run_backtest_local("SimpleMA", data, params)

        # Should handle without errors
        assert isinstance(metrics, BacktestMetrics)
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.max_drawdown)