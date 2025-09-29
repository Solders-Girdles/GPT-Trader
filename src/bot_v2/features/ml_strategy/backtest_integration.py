"""
Integration with backtesting for ML-driven strategy selection - LOCAL to this slice.
"""

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from .types import StrategyName, StrategyPrediction


def run_ml_backtest(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    rebalance_frequency: int,
    min_confidence: float,
    predictor: Callable[[str, int, int], list[StrategyPrediction]],
) -> dict:
    """
    Run backtest with ML-driven dynamic strategy selection.

    Switches strategies based on ML predictions at rebalance intervals.
    """
    # Fetch historical data (simplified)
    from .market_data import fetch_market_data

    lookback_days = (end_date - start_date).days
    data = fetch_market_data(symbol, lookback_days)

    # Initialize backtest state
    capital = initial_capital
    positions = 0
    equity_curve = [capital]
    returns = []
    strategy_changes = 0
    total_confidence = 0
    confidence_count = 0
    trades = []
    current_strategy = None
    previous_strategy = None
    entry_price: float | None = None

    # Run backtest with periodic rebalancing
    for i in range(rebalance_frequency, len(data), rebalance_frequency):
        # Get ML prediction for current period
        predictions = predictor(symbol, 30, 1)  # Look back 30 days, get top 1

        if predictions and predictions[0].confidence >= min_confidence:
            recommended_strategy = predictions[0].strategy
            confidence = predictions[0].confidence

            # Track confidence
            total_confidence += confidence
            confidence_count += 1

            # Check for strategy change
            if recommended_strategy != current_strategy:
                previous_strategy = current_strategy
                current_strategy = recommended_strategy
                strategy_changes += 1

                # Close existing position if any
                if positions != 0 and entry_price is not None and previous_strategy is not None:
                    exit_price = data["close"].iloc[i]
                    trade_return = (exit_price - entry_price) / entry_price * positions
                    capital *= 1 + trade_return
                    trades.append(
                        {
                            "strategy": str(previous_strategy),
                            "entry": entry_price,
                            "exit": exit_price,
                            "return": trade_return,
                        }
                    )
                    positions = 0

            # Execute strategy (simplified)
            if positions == 0:
                # Enter new position based on strategy
                signal = _get_strategy_signal(current_strategy, data.iloc[max(0, i - 30) : i + 1])

                if signal != 0:
                    entry_price = data["close"].iloc[i]
                    positions = signal  # 1 for long, -1 for short

        # Update equity
        if positions != 0 and entry_price is not None:
            current_price = data["close"].iloc[i]
            unrealized_pnl = (current_price - entry_price) / entry_price * positions
            current_equity = capital * (1 + unrealized_pnl)
        else:
            current_equity = capital

        equity_curve.append(current_equity)

        # Calculate period return
        period_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
        returns.append(period_return)

    # Close final position
    if positions != 0 and entry_price is not None and current_strategy is not None:
        exit_price = data["close"].iloc[-1]
        trade_return = (exit_price - entry_price) / entry_price * positions
        capital *= 1 + trade_return
        trades.append(
            {
                "strategy": str(current_strategy),
                "entry": entry_price,
                "exit": exit_price,
                "return": trade_return,
            }
        )

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital

    if returns:
        sharpe_ratio = calculate_sharpe(returns)
        max_drawdown = calculate_max_drawdown(equity_curve)
    else:
        sharpe_ratio = 0
        max_drawdown = 0

    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "strategy_changes": strategy_changes,
        "avg_confidence": avg_confidence,
        "final_capital": capital,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _get_strategy_signal(strategy: StrategyName, data: pd.DataFrame) -> int:
    """
    Get trading signal from strategy.

    Returns 1 for long, -1 for short, 0 for no position.
    """
    if len(data) < 30:
        return 0

    close_prices = data["close"].values

    if strategy == StrategyName.SIMPLE_MA:
        # Simple MA crossover
        ma_fast = np.mean(close_prices[-10:])
        ma_slow = np.mean(close_prices[-30:])

        if ma_fast > ma_slow * 1.01:  # 1% threshold
            return 1
        elif ma_fast < ma_slow * 0.99:
            return -1
        return 0

    elif strategy == StrategyName.MOMENTUM:
        # Momentum strategy
        momentum = close_prices[-1] / close_prices[-20] - 1

        if momentum > 0.05:  # 5% momentum
            return 1
        elif momentum < -0.05:
            return -1
        return 0

    elif strategy == StrategyName.MEAN_REVERSION:
        # Mean reversion
        mean = np.mean(close_prices[-20:])
        std = np.std(close_prices[-20:])
        current = close_prices[-1]

        z_score = (current - mean) / std if std > 0 else 0

        if z_score < -2:  # Oversold
            return 1
        elif z_score > 2:  # Overbought
            return -1
        return 0

    elif strategy == StrategyName.VOLATILITY:
        # Volatility breakout
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

        if volatility > 0.02:  # High volatility
            # Trade in direction of recent move
            recent_return = close_prices[-1] / close_prices[-5] - 1
            if recent_return > 0:
                return 1
            else:
                return -1
        return 0

    elif strategy == StrategyName.BREAKOUT:
        # Breakout strategy
        high_20 = np.max(close_prices[-20:])
        low_20 = np.min(close_prices[-20:])
        current = close_prices[-1]

        if current > high_20 * 0.99:  # Near 20-day high
            return 1
        elif current < low_20 * 1.01:  # Near 20-day low
            return -1
        return 0

    return 0


def calculate_sharpe(returns: list[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate

    if len(excess_returns) < 2:
        return 0.0

    mean_excess = np.mean(excess_returns)
    std_returns = np.std(excess_returns)

    if std_returns == 0:
        return 0.0

    # Annualize
    sharpe = (mean_excess / std_returns) * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0

    for value in equity_curve[1:]:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

    return -max_dd  # Return as negative percentage
