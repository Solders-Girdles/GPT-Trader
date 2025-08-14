"""
Parallel Optimization Engine

Provides multiprocessing capabilities for strategy optimization:
- Parallel parameter sweeps
- Distributed backtesting
- Intelligent work distribution
- Results aggregation and analysis
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from bot.config import get_config


@dataclass
class OptimizationConfig:
    """Configuration for parallel optimization"""

    strategy_class: type
    parameter_grid: dict[str, list[Any]]
    data: pd.DataFrame
    initial_cash: float = None  # Will use config default if not specified
    commission: float = None  # Will use config default if not specified
    max_workers: int | None = None

    def __post_init__(self):
        """Load defaults from unified configuration if not specified."""
        config = get_config()
        if self.initial_cash is None:
            self.initial_cash = float(config.financial.capital.backtesting_capital)
        if self.commission is None:
            self.commission = config.financial.costs.commission_rate_decimal

    chunk_size: int = 1
    timeout: float | None = None

    # Optimization constraints
    min_trades: int = 10
    max_drawdown_limit: float = 0.5
    min_sharpe_ratio: float = 0.0


@dataclass
class OptimizationResult:
    """Single optimization result"""

    parameters: dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    execution_time: float
    error: str | None = None


class ParallelOptimizer:
    """
    High-performance parallel optimizer for strategy parameters.

    Features:
    - Multiprocessing for CPU-intensive optimization
    - Intelligent parameter grid exploration
    - Result caching and persistence
    - Progress monitoring and early stopping
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
        self._results_cache = {}

    def optimize_parameters(self, config: OptimizationConfig) -> list[OptimizationResult]:
        """
        Run parallel parameter optimization.

        Args:
            config: Optimization configuration

        Returns:
            List of optimization results sorted by performance
        """
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(config.parameter_grid)
        total_combinations = len(param_combinations)

        self.logger.info("🚀 Starting parallel optimization")
        self.logger.info(f"   📊 Total combinations: {total_combinations:,}")
        self.logger.info(f"   ⚡ Workers: {self.max_workers}")
        self.logger.info(f"   🎯 Chunk size: {config.chunk_size}")

        # Create partial function for worker
        worker_func = partial(
            self._optimize_single_combination,
            strategy_class=config.strategy_class,
            data=config.data,
            initial_cash=config.initial_cash,
            commission=config.commission,
        )

        # Execute optimization in parallel
        results = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(worker_func, params): params for params in param_combinations
            }

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_params, timeout=config.timeout):
                try:
                    result = future.result()
                    if self._is_valid_result(result, config):
                        results.append(result)

                    completed_count += 1
                    if completed_count % max(1, total_combinations // 20) == 0:
                        progress = completed_count / total_combinations
                        elapsed = time.time() - start_time
                        eta = elapsed / progress - elapsed if progress > 0 else 0

                        self.logger.info(
                            f"   📈 Progress: {completed_count}/{total_combinations} "
                            f"({progress:.1%}) | ETA: {eta:.1f}s"
                        )

                except Exception as e:
                    params = future_to_params[future]
                    self.logger.warning(f"   ⚠️  Failed optimization for {params}: {e}")

        total_time = time.time() - start_time

        # Sort results by performance (configurable metric)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        self.logger.info("✅ Optimization complete!")
        self.logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
        self.logger.info(f"   🎯 Valid results: {len(results)}/{total_combinations}")
        self.logger.info(f"   🚀 Throughput: {total_combinations/total_time:.1f} combinations/sec")

        if results:
            best = results[0]
            self.logger.info(
                f"   🏆 Best result: Sharpe={best.sharpe_ratio:.3f}, Return={best.total_return:.2%}"
            )

        return results

    def optimize_with_adaptive_grid(
        self, config: OptimizationConfig, max_iterations: int = 3, refinement_factor: float = 0.5
    ) -> list[OptimizationResult]:
        """
        Adaptive grid optimization that refines search space iteratively.

        Args:
            config: Initial optimization configuration
            max_iterations: Maximum refinement iterations
            refinement_factor: How much to refine grid each iteration (0.5 = half the range)

        Returns:
            Final optimization results
        """
        current_config = config
        all_results = []

        for iteration in range(max_iterations):
            self.logger.info(f"🔄 Adaptive iteration {iteration + 1}/{max_iterations}")

            # Run optimization
            results = self.optimize_parameters(current_config)
            all_results.extend(results)

            if not results:
                self.logger.warning("No valid results in iteration, stopping adaptive optimization")
                break

            # Find best results for refinement
            top_results = results[: min(5, len(results))]

            if iteration < max_iterations - 1:
                # Refine grid around best results
                current_config = self._refine_parameter_grid(config, top_results, refinement_factor)

        # Return all unique results sorted by performance
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        return unique_results

    def benchmark_parallelization(self, config: OptimizationConfig) -> dict[str, Any]:
        """
        Benchmark different parallelization configurations.

        Returns:
            Performance metrics for different worker counts
        """
        param_combinations = self._generate_parameter_combinations(config.parameter_grid)
        sample_size = min(50, len(param_combinations))  # Benchmark with subset
        sample_combinations = param_combinations[:sample_size]

        results = {}

        for workers in [1, 2, 4, self.max_workers]:
            if workers > self.max_workers:
                continue

            self.logger.info(f"🧪 Benchmarking with {workers} workers")

            start_time = time.time()

            # Create temporary config for benchmark
            OptimizationConfig(
                strategy_class=config.strategy_class,
                parameter_grid={},  # Will use sample_combinations directly
                data=config.data,
                initial_cash=config.initial_cash,
                commission=config.commission,
                max_workers=workers,
            )

            worker_func = partial(
                self._optimize_single_combination,
                strategy_class=config.strategy_class,
                data=config.data,
                initial_cash=config.initial_cash,
                commission=config.commission,
            )

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(worker_func, params) for params in sample_combinations]

                valid_results = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result.error is None:
                            valid_results += 1
                    except Exception:
                        pass

            total_time = time.time() - start_time
            throughput = sample_size / total_time

            results[workers] = {
                "workers": workers,
                "sample_size": sample_size,
                "execution_time": total_time,
                "throughput": throughput,
                "valid_results": valid_results,
                "efficiency": throughput / workers,  # Results per worker per second
            }

            self.logger.info(f"   ⚡ Throughput: {throughput:.1f} combinations/sec")

        return results

    @staticmethod
    def _optimize_single_combination(
        params: dict[str, Any],
        strategy_class: type,
        data: pd.DataFrame,
        initial_cash: float,
        commission: float,
    ) -> OptimizationResult:
        """
        Optimize a single parameter combination.

        This function runs in a separate process.
        """
        start_time = time.time()

        try:
            # Create strategy instance based on strategy_class type
            if "TALibMAParams" in str(strategy_class) or "OptimizedMAParams" in str(strategy_class):
                # Handle parameter dataclass case - create params object then strategy
                from bot.strategy.optimized_ma import OptimizedMAStrategy
                from bot.strategy.talib_optimized_ma import TALibOptimizedMAStrategy

                if "TALibMAParams" in str(strategy_class):
                    param_obj = strategy_class(**params)
                    strategy = TALibOptimizedMAStrategy(param_obj)
                else:
                    param_obj = strategy_class(**params)
                    strategy = OptimizedMAStrategy(param_obj)
            else:
                # Handle direct strategy class case
                strategy = strategy_class(**params)

            # Generate signals
            signals = strategy.generate_signals(data)

            # Calculate basic performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = returns * signals["signal"].shift(1)

            # Basic metrics calculation
            total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            sharpe_ratio = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                if strategy_returns.std() > 0
                else 0
            )

            # Drawdown calculation
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Trade counting (simplified)
            position_changes = signals["signal"].diff().abs()
            num_trades = int(position_changes.sum() / 2)  # Round trips

            # Win rate calculation (simplified)
            trade_returns = strategy_returns[position_changes > 0]
            win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0

            # Profit factor (simplified)
            positive_returns = trade_returns[trade_returns > 0].sum()
            negative_returns = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = (
                positive_returns / negative_returns if negative_returns > 0 else float("inf")
            )

            execution_time = time.time() - start_time

            return OptimizationResult(
                parameters=params.copy(),
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                num_trades=num_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return OptimizationResult(
                parameters=params.copy(),
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                num_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                execution_time=execution_time,
                error=str(e),
            )

    def _generate_parameter_combinations(
        self, parameter_grid: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination, strict=False))
            combinations.append(param_dict)

        return combinations

    def _is_valid_result(self, result: OptimizationResult, config: OptimizationConfig) -> bool:
        """Check if result meets minimum criteria"""
        if result.error is not None:
            return False

        if result.num_trades < config.min_trades:
            return False

        if abs(result.max_drawdown) > config.max_drawdown_limit:
            return False

        if result.sharpe_ratio < config.min_sharpe_ratio:
            return False

        return True

    def _refine_parameter_grid(
        self,
        original_config: OptimizationConfig,
        best_results: list[OptimizationResult],
        refinement_factor: float,
    ) -> OptimizationConfig:
        """Refine parameter grid around best results"""
        # Calculate refined ranges for each parameter
        refined_grid = {}

        for param_name in original_config.parameter_grid.keys():
            # Get values from best results
            best_values = [result.parameters[param_name] for result in best_results]

            # Calculate new range
            min_val = min(best_values)
            max_val = max(best_values)

            if min_val == max_val:
                # If all best results have same value, keep it
                refined_grid[param_name] = [min_val]
            else:
                # Create refined range
                range_size = max_val - min_val
                expanded_range = range_size * (1 + refinement_factor)

                new_min = max(
                    min_val - expanded_range * 0.5, min(original_config.parameter_grid[param_name])
                )
                new_max = min(
                    max_val + expanded_range * 0.5, max(original_config.parameter_grid[param_name])
                )

                # Generate new values
                if isinstance(best_values[0], int):
                    step = max(1, int(expanded_range / 5))
                    refined_grid[param_name] = list(range(int(new_min), int(new_max) + 1, step))
                else:
                    num_steps = 5
                    step = (new_max - new_min) / (num_steps - 1)
                    refined_grid[param_name] = [new_min + i * step for i in range(num_steps)]

        return OptimizationConfig(
            strategy_class=original_config.strategy_class,
            parameter_grid=refined_grid,
            data=original_config.data,
            initial_cash=original_config.initial_cash,
            commission=original_config.commission,
            max_workers=original_config.max_workers,
            chunk_size=original_config.chunk_size,
            timeout=original_config.timeout,
            min_trades=original_config.min_trades,
            max_drawdown_limit=original_config.max_drawdown_limit,
            min_sharpe_ratio=original_config.min_sharpe_ratio,
        )

    def _deduplicate_results(self, results: list[OptimizationResult]) -> list[OptimizationResult]:
        """Remove duplicate results based on parameters"""
        seen_params = set()
        unique_results = []

        for result in results:
            param_key = tuple(sorted(result.parameters.items()))
            if param_key not in seen_params:
                seen_params.add(param_key)
                unique_results.append(result)

        return unique_results


def benchmark_multiprocessing():
    """Benchmark multiprocessing performance with TA-Lib strategy"""

    # Ensure imports work
    try:
        from ..strategy.talib_optimized_ma import TALibMAParams, TALibOptimizedMAStrategy
    except ImportError:
        print("⚠️  TA-Lib strategy not available for benchmark")
        return

    print("🚀 Multiprocessing Benchmark")
    print("=" * 50)

    # Generate test data
    np.random.seed(42)
    n_days = 2000  # Reasonable size for benchmark
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    returns = np.random.normal(0.0008, 0.018, n_days)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15.5, 0.35, n_days).astype(int)

    data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    # Define parameter grid
    parameter_grid = {
        "fast": [5, 8, 10, 12, 15],
        "slow": [20, 25, 30, 35, 40],
        "volume_filter": [True, False],
        "rsi_filter": [True, False],
    }

    # Create optimization config
    config = OptimizationConfig(
        strategy_class=TALibMAParams,  # Use parameter class
        parameter_grid=parameter_grid,
        data=data,
        initial_cash=100000,
        commission=0.001,
        min_trades=5,
    )

    # Run benchmark
    optimizer = ParallelOptimizer()
    benchmark_results = optimizer.benchmark_parallelization(config)

    print("\n📊 PARALLELIZATION BENCHMARK:")
    print(
        f"   📈 Parameter combinations: {len(optimizer._generate_parameter_combinations(parameter_grid))}"
    )
    print(f"   📊 Data size: {len(data):,} rows")

    for workers, metrics in benchmark_results.items():
        speedup = (
            metrics["throughput"] / benchmark_results[1]["throughput"]
            if 1 in benchmark_results
            else 1.0
        )
        print(f"\n   ⚡ {workers} workers:")
        print(f"      🚀 Throughput: {metrics['throughput']:.1f} combinations/sec")
        print(f"      📈 Speedup: {speedup:.1f}x")
        print(f"      ⚙️  Efficiency: {metrics['efficiency']:.1f} combinations/worker/sec")
        print(
            f"      ✅ Success rate: {metrics['valid_results']}/{metrics['sample_size']} ({metrics['valid_results']/metrics['sample_size']:.1%})"
        )

    return benchmark_results


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_multiprocessing()
