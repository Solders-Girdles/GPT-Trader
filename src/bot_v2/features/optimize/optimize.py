"""
Main optimization orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from bot_v2.data_providers import get_data_provider
from bot_v2.features.optimize.backtester import run_backtest_local
from bot_v2.features.optimize.strategies import get_strategy_params
from bot_v2.features.optimize.types import (
    OptimizationResult,
    ParameterGrid,
    WalkForwardResult,
    WalkForwardWindow,
)


def optimize_strategy(
    strategy: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    param_grid: dict[str, list[Any]] | None = None,
    metric: str = "sharpe_ratio",
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> OptimizationResult:
    """
    Optimize strategy parameters.

    Args:
        strategy: Strategy name to optimize
        symbol: Stock symbol
        start_date: Start of optimization period
        end_date: End of optimization period
        param_grid: Parameters to optimize (None for defaults)
        metric: Metric to optimize ('sharpe_ratio', 'return', 'calmar')
        commission: Commission rate
        slippage: Slippage rate

    Returns:
        Optimization result with best parameters
    """
    start_time = time.time()

    # Get default parameter grid if not provided
    if param_grid is None:
        param_grid = get_strategy_params(strategy)

    # Create parameter grid
    grid = ParameterGrid(strategy=strategy, parameters=param_grid)

    print(f"Optimizing {strategy} on {symbol}")
    print(f"Testing {grid.total_combinations()} parameter combinations")

    # Fetch data once
    data = fetch_data(symbol, start_date, end_date)

    # Test all combinations
    all_results = []
    best_score = -float("inf")
    best_params = None
    best_metrics = None

    for i, params in enumerate(grid.get_combinations()):
        # Run backtest with these parameters
        metrics = run_backtest_local(strategy, data, params, commission, slippage)

        # Get optimization metric
        if metric == "sharpe_ratio":
            score = metrics.sharpe_ratio
        elif metric == "return":
            score = metrics.total_return
        elif metric == "calmar":
            score = metrics.calmar_ratio
        else:
            score = metrics.sharpe_ratio

        # Track results
        result = {"params": params, "metrics": metrics, "score": score}
        all_results.append(result)

        # Update best
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{grid.total_combinations()} combinations tested")

    optimization_time = time.time() - start_time

    print(f"\nOptimization complete in {optimization_time:.1f} seconds")
    print(f"Best {metric}: {best_score:.3f}")
    print(f"Best params: {best_params}")

    return OptimizationResult(
        strategy=strategy,
        symbol=symbol,
        period=(start_date, end_date),
        best_params=best_params,
        best_metrics=best_metrics,
        all_results=all_results,
        optimization_time=optimization_time,
    )


def grid_search(
    strategies: list[str],
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    metric: str = "sharpe_ratio",
) -> dict[str, OptimizationResult]:
    """
    Perform grid search across multiple strategies.

    Args:
        strategies: List of strategies to optimize
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        metric: Metric to optimize

    Returns:
        Dict of strategy -> optimization result
    """
    results = {}

    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Optimizing {strategy}")
        print("=" * 50)

        result = optimize_strategy(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
        )

        results[strategy] = result

    # Find best overall strategy
    best_strategy = None
    best_score = -float("inf")

    for strategy, result in results.items():
        score = getattr(result.best_metrics, metric)
        if score > best_score:
            best_score = score
            best_strategy = strategy

    print(f"\n{'='*50}")
    print(f"BEST STRATEGY: {best_strategy}")
    print(f"Best {metric}: {best_score:.3f}")
    print("=" * 50)

    return results


def walk_forward_analysis(
    strategy: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    window_size: int = 60,  # Training window in days
    step_size: int = 30,  # Step forward in days
    test_size: int = 30,  # Test window in days
    param_grid: dict[str, list[Any]] | None = None,
) -> WalkForwardResult:
    """
    Perform walk-forward optimization.

    Args:
        strategy: Strategy to optimize
        symbol: Stock symbol
        start_date: Start of analysis period
        end_date: End of analysis period
        window_size: Training window size in days
        step_size: Days to step forward each iteration
        test_size: Test window size in days
        param_grid: Parameters to optimize

    Returns:
        Walk-forward analysis result
    """
    print(f"Walk-Forward Analysis for {strategy} on {symbol}")
    print(f"Training window: {window_size} days")
    print(f"Test window: {test_size} days")
    print(f"Step size: {step_size} days")

    # Generate windows
    windows = []
    current_start = start_date

    while current_start + timedelta(days=window_size + test_size) <= end_date:
        train_start = current_start
        train_end = current_start + timedelta(days=window_size)
        test_start = train_end
        test_end = test_start + timedelta(days=test_size)

        print(f"\nWindow {len(windows) + 1}:")
        print(f"  Train: {train_start.date()} to {train_end.date()}")
        print(f"  Test: {test_start.date()} to {test_end.date()}")

        # Optimize on training period
        opt_result = optimize_strategy(
            strategy=strategy,
            symbol=symbol,
            start_date=train_start,
            end_date=train_end,
            param_grid=param_grid,
            metric="sharpe_ratio",
        )

        # Test on out-of-sample period
        test_data = fetch_data(symbol, test_start, test_end)
        test_metrics = run_backtest_local(
            strategy, test_data, opt_result.best_params, commission=0.001, slippage=0.0005
        )

        # Create window result
        window = WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=opt_result.best_params,
            train_metrics=opt_result.best_metrics,
            test_metrics=test_metrics,
        )

        windows.append(window)

        print(f"  Train Return: {window.train_metrics.total_return:.2%}")
        print(f"  Test Return: {window.test_metrics.total_return:.2%}")
        print(f"  Efficiency: {window.get_efficiency():.2f}")

        # Move to next window
        current_start += timedelta(days=step_size)

    # Calculate overall metrics
    efficiencies = [w.get_efficiency() for w in windows]
    avg_efficiency = np.mean(efficiencies) if efficiencies else 0

    # Consistency score (inverse of std dev of returns)
    test_returns = [w.test_metrics.total_return for w in windows]
    if len(test_returns) > 1:
        consistency_score = 1 / (1 + np.std(test_returns))
    else:
        consistency_score = 0.5

    # Robustness score (combination of efficiency and consistency)
    robustness_score = avg_efficiency * 0.6 + consistency_score * 0.4
    robustness_score = min(1.0, max(0.0, robustness_score))

    print(f"\n{'='*50}")
    print("Walk-Forward Complete")
    print(f"Windows analyzed: {len(windows)}")
    print(f"Average efficiency: {avg_efficiency:.2f}")
    print(f"Consistency: {consistency_score:.2%}")
    print(f"Robustness: {robustness_score:.2%}")

    return WalkForwardResult(
        strategy=strategy,
        symbol=symbol,
        windows=windows,
        avg_efficiency=avg_efficiency,
        consistency_score=consistency_score,
        robustness_score=robustness_score,
    )


# Helper functions


def fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch historical data for optimization."""
    provider = get_data_provider()
    data = provider.get_historical_data(
        symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )

    if data.empty:
        raise ValueError(f"No data available for {symbol}")

    # Standardize columns
    data.columns = data.columns.str.lower()

    return data
