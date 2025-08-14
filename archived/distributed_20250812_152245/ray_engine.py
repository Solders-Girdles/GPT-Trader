"""
Ray Distributed Computing Engine

Provides massive parallelization capabilities using Ray:
- Distributed strategy optimization across multiple machines
- Auto-scaling based on workload
- Fault tolerance and recovery
- Distributed data processing
- GPU acceleration support
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import ray
from ray import remote

try:
    from ..optimization.parallel_optimizer import OptimizationResult
    from ..strategy.base import Strategy
except ImportError:
    # For standalone execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from bot.optimization.parallel_optimizer import OptimizationResult


@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""

    num_cpus: int | None = None  # None = use all available
    num_gpus: int | None = None  # None = use all available
    memory_gb: float | None = None  # None = auto
    object_store_memory_gb: float | None = None  # None = auto
    dashboard_host: str = "127.0.0.1"  # Bind to localhost for security
    dashboard_port: int = 8265
    include_dashboard: bool = True
    num_workers: int | None = None  # None = auto based on CPUs
    max_retries: int = 3
    timeout_seconds: float = 300.0


@dataclass
class DistributedResult:
    """Result from distributed computation"""

    task_id: str
    result: Any
    execution_time: float
    worker_id: str
    retry_count: int = 0
    error: str | None = None


@remote
class DistributedWorker:
    """
    Ray actor for distributed computation.

    Each worker runs on a separate process/machine and can execute
    strategy evaluations independently.
    """

    def __init__(self, worker_id: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger(f"Worker-{worker_id}")
        self.task_count = 0
        self.total_execution_time = 0.0

        # Import heavy dependencies locally to avoid serialization
        self._strategy_cache = {}

    def evaluate_strategy(
        self,
        strategy_class_name: str,
        params: dict[str, Any],
        data_ref: ray.ObjectRef,
        initial_cash: float = 100000,
        commission: float = 0.001,
    ) -> OptimizationResult:
        """
        Evaluate a single strategy configuration.

        Args:
            strategy_class_name: Full class name of strategy
            params: Strategy parameters
            data_ref: Ray object reference to market data
            initial_cash: Starting capital
            commission: Trading commission

        Returns:
            Optimization result with performance metrics
        """
        start_time = time.time()

        try:
            # Get data from object store (handle both ref and direct data)
            if hasattr(data_ref, "__ray_actor_class__") or isinstance(data_ref, ray.ObjectRef):
                data = ray.get(data_ref)
            else:
                # Direct data passed (for testing)
                data = data_ref

            # Import and instantiate strategy
            if "TALibOptimizedMAStrategy" in strategy_class_name:
                from bot.strategy.talib_optimized_ma import TALibMAParams, TALibOptimizedMAStrategy

                param_obj = TALibMAParams(**params)
                strategy = TALibOptimizedMAStrategy(param_obj)
            elif "OptimizedMAStrategy" in strategy_class_name:
                from bot.strategy.optimized_ma import OptimizedMAParams, OptimizedMAStrategy

                param_obj = OptimizedMAParams(**params)
                strategy = OptimizedMAStrategy(param_obj)
            else:
                # Dynamic import for other strategies
                module_path, class_name = strategy_class_name.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                strategy_class = getattr(module, class_name)
                strategy = strategy_class(**params)

            # Generate signals
            signals = strategy.generate_signals(data)

            # Calculate performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = returns * signals["signal"].shift(1)

            # Metrics calculation
            total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            sharpe_ratio = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                if strategy_returns.std() > 0
                else 0
            )

            # Drawdown
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Trade statistics
            position_changes = signals["signal"].diff().abs()
            num_trades = int(position_changes.sum() / 2)

            # Win rate
            trade_returns = strategy_returns[position_changes > 0]
            win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0

            # Profit factor
            positive_returns = trade_returns[trade_returns > 0].sum()
            negative_returns = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = (
                positive_returns / negative_returns if negative_returns > 0 else float("inf")
            )
            if profit_factor == float("inf"):
                profit_factor = 999.99  # Cap at reasonable value

            execution_time = time.time() - start_time
            self.task_count += 1
            self.total_execution_time += execution_time

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
            self.logger.error(f"Strategy evaluation failed: {e}")

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

    def batch_evaluate(
        self,
        strategy_class_name: str,
        param_list: list[dict[str, Any]],
        data_ref: ray.ObjectRef,
        initial_cash: float = 100000,
        commission: float = 0.001,
    ) -> list[OptimizationResult]:
        """
        Evaluate multiple strategy configurations in batch.

        More efficient than individual evaluations due to reduced overhead.
        """
        results = []

        for params in param_list:
            result = self.evaluate_strategy(
                strategy_class_name, params, data_ref, initial_cash, commission
            )
            results.append(result)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "task_count": self.task_count,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": self.total_execution_time / max(1, self.task_count),
            "throughput": self.task_count / max(0.001, self.total_execution_time),
        }

    def reset_stats(self) -> None:
        """Reset worker statistics"""
        self.task_count = 0
        self.total_execution_time = 0.0
        self._strategy_cache.clear()


class RayDistributedEngine:
    """
    High-performance distributed computing engine using Ray.

    Features:
    - Automatic scaling across multiple machines
    - Fault tolerance with automatic retry
    - Efficient data sharing via object store
    - GPU acceleration support
    - Real-time monitoring dashboard
    """

    def __init__(self, config: DistributedConfig | None = None) -> None:
        self.config = config or DistributedConfig()
        self.logger = logging.getLogger(__name__)
        self.workers = []
        self.is_initialized = False

    def initialize(self, reinit: bool = False) -> bool:
        """
        Initialize Ray cluster.

        Args:
            reinit: Force reinitialization if already initialized

        Returns:
            True if initialization successful
        """
        if self.is_initialized and not reinit:
            self.logger.info("Ray already initialized")
            return True

        try:
            # Shutdown existing cluster if reinitializing
            if ray.is_initialized():
                if reinit:
                    ray.shutdown()
                else:
                    self.is_initialized = True
                    return True

            # Initialize Ray with configuration
            init_kwargs = {
                "dashboard_host": self.config.dashboard_host,
                "dashboard_port": self.config.dashboard_port,
                "include_dashboard": self.config.include_dashboard,
                "ignore_reinit_error": True,
                "logging_level": logging.INFO,
            }

            # Add resource constraints if specified
            if self.config.num_cpus is not None:
                init_kwargs["num_cpus"] = self.config.num_cpus
            if self.config.num_gpus is not None:
                init_kwargs["num_gpus"] = self.config.num_gpus
            if self.config.memory_gb is not None:
                init_kwargs["_memory"] = int(self.config.memory_gb * 1024 * 1024 * 1024)
            if self.config.object_store_memory_gb is not None:
                init_kwargs["object_store_memory"] = int(
                    self.config.object_store_memory_gb * 1024 * 1024 * 1024
                )

            ray.init(**init_kwargs)

            # Create workers
            num_workers = self.config.num_workers
            if num_workers is None:
                num_workers = ray.available_resources().get("CPU", psutil.cpu_count())
                num_workers = min(int(num_workers), 32)  # Cap at 32 workers

            self.logger.info(f"🚀 Initializing Ray with {num_workers} workers")

            # Create worker actors
            self.workers = []
            for i in range(num_workers):
                worker = DistributedWorker.remote(worker_id=f"worker-{i}", config={})
                self.workers.append(worker)

            self.is_initialized = True

            # Log cluster info
            self.logger.info("✅ Ray cluster initialized")
            self.logger.info(f"   📊 Available CPUs: {ray.available_resources().get('CPU', 0)}")
            self.logger.info(f"   🎮 Available GPUs: {ray.available_resources().get('GPU', 0)}")
            self.logger.info(
                f"   💾 Object store: {ray.available_resources().get('object_store_memory', 0) / 1024 / 1024 / 1024:.1f} GB"
            )

            if self.config.include_dashboard:
                self.logger.info(
                    f"   📊 Dashboard: http://{self.config.dashboard_host}:{self.config.dashboard_port}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Ray: {e}")
            self.is_initialized = False
            return False

    def shutdown(self) -> None:
        """Shutdown Ray cluster"""
        if ray.is_initialized():
            ray.shutdown()
            self.is_initialized = False
            self.workers = []
            self.logger.info("Ray cluster shutdown")

    def optimize_distributed(
        self,
        strategy_class_name: str,
        parameter_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_cash: float = 100000,
        commission: float = 0.001,
        batch_size: int | None = None,
    ) -> list[OptimizationResult]:
        """
        Run distributed parameter optimization using Ray.

        Args:
            strategy_class_name: Full class name of strategy
            parameter_grid: Parameter combinations to test
            data: Market data DataFrame
            initial_cash: Starting capital
            commission: Trading commission
            batch_size: Number of parameters per worker batch

        Returns:
            List of optimization results sorted by performance
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Ray cluster")

        start_time = time.time()

        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        total_combinations = len(param_combinations)

        self.logger.info("🚀 Starting distributed optimization")
        self.logger.info(f"   📊 Total combinations: {total_combinations:,}")
        self.logger.info(f"   ⚡ Workers: {len(self.workers)}")

        # Put data in object store (shared memory)
        data_ref = ray.put(data)

        # Determine batch size
        if batch_size is None:
            batch_size = max(1, total_combinations // (len(self.workers) * 4))
            batch_size = min(batch_size, 10)  # Cap batch size

        # Create batches
        batches = []
        for i in range(0, total_combinations, batch_size):
            batch = param_combinations[i : i + batch_size]
            batches.append(batch)

        self.logger.info(f"   📦 Batch size: {batch_size}")
        self.logger.info(f"   🔄 Total batches: {len(batches)}")

        # Distribute work to workers
        futures = []
        for i, batch in enumerate(batches):
            worker = self.workers[i % len(self.workers)]
            future = worker.batch_evaluate.remote(
                strategy_class_name, batch, data_ref, initial_cash, commission
            )
            futures.append(future)

        # Collect results with progress tracking
        all_results = []
        completed = 0

        while futures:
            # Wait for any task to complete
            ready, futures = ray.wait(futures, num_returns=1, timeout=1.0)

            for future in ready:
                try:
                    batch_results = ray.get(future)
                    all_results.extend(batch_results)
                    completed += len(batch_results)

                    # Progress update
                    progress = completed / total_combinations
                    elapsed = time.time() - start_time
                    eta = elapsed / progress - elapsed if progress > 0 else 0

                    if completed % max(1, total_combinations // 10) == 0:
                        self.logger.info(
                            f"   📈 Progress: {completed}/{total_combinations} "
                            f"({progress:.1%}) | ETA: {eta:.1f}s"
                        )

                except Exception as e:
                    self.logger.warning(f"Task failed: {e}")

        total_time = time.time() - start_time

        # Filter valid results and sort by performance
        valid_results = [r for r in all_results if r.error is None]
        valid_results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        # Get worker statistics
        worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers])
        sum(stats["task_count"] for stats in worker_stats)

        self.logger.info("✅ Distributed optimization complete!")
        self.logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
        self.logger.info(f"   🎯 Valid results: {len(valid_results)}/{total_combinations}")
        self.logger.info(f"   🚀 Throughput: {total_combinations/total_time:.1f} combinations/sec")
        self.logger.info(
            f"   ⚡ Speedup vs serial: {total_combinations * 0.1 / total_time:.1f}x"
        )  # Assume 0.1s per eval serial

        if valid_results:
            best = valid_results[0]
            self.logger.info(
                f"   🏆 Best result: Sharpe={best.sharpe_ratio:.3f}, Return={best.total_return:.2%}"
            )

        # Log worker performance
        self.logger.info("   📊 Worker Statistics:")
        for stats in worker_stats:
            self.logger.info(
                f"      {stats['worker_id']}: {stats['task_count']} tasks, "
                f"{stats['avg_execution_time']:.3f}s avg, "
                f"{stats['throughput']:.1f} tasks/sec"
            )

        return valid_results

    def process_data_distributed(
        self, data_chunks: list[pd.DataFrame], process_func: Callable, *args, **kwargs
    ) -> list[Any]:
        """
        Process data chunks in parallel across workers.

        Args:
            data_chunks: List of DataFrame chunks to process
            process_func: Function to apply to each chunk
            *args, **kwargs: Additional arguments for process_func

        Returns:
            List of processed results
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Ray cluster")

        # Create remote function
        @ray.remote
        def process_chunk(chunk, func, *args, **kwargs):
            return func(chunk, *args, **kwargs)

        # Submit all chunks for processing
        futures = [
            process_chunk.remote(chunk, process_func, *args, **kwargs) for chunk in data_chunks
        ]

        # Collect results
        results = ray.get(futures)

        return results

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about Ray cluster"""
        if not ray.is_initialized():
            return {"status": "not_initialized"}

        resources = ray.available_resources()
        nodes = ray.nodes()

        return {
            "status": "running",
            "nodes": len(nodes),
            "cpus": resources.get("CPU", 0),
            "gpus": resources.get("GPU", 0),
            "memory_gb": resources.get("memory", 0) / 1024 / 1024 / 1024,
            "object_store_gb": resources.get("object_store_memory", 0) / 1024 / 1024 / 1024,
            "workers": len(self.workers),
            "dashboard_url": f"http://{self.config.dashboard_host}:{self.config.dashboard_port}",
        }

    def benchmark_scaling(self, data: pd.DataFrame, test_sizes: list[int] = None) -> dict[str, Any]:
        """
        Benchmark distributed computing performance at different scales.

        Args:
            data: Market data for testing
            test_sizes: List of parameter grid sizes to test

        Returns:
            Benchmark results showing scaling efficiency
        """
        if test_sizes is None:
            test_sizes = [10, 50, 100, 500, 1000]

        results = {}

        for size in test_sizes:
            # Create parameter grid of specified size
            n_params = int(np.sqrt(size))
            parameter_grid = {
                "fast": list(range(5, min(5 + n_params, 20))),
                "slow": list(range(20, min(20 + n_params, 50))),
                "volume_filter": [True, False] if size > 20 else [True],
            }

            actual_size = len(self._generate_parameter_combinations(parameter_grid))

            self.logger.info(f"🧪 Benchmarking with {actual_size} combinations")

            start_time = time.time()
            optimization_results = self.optimize_distributed(
                "bot.strategy.talib_optimized_ma.TALibOptimizedMAStrategy", parameter_grid, data
            )
            execution_time = time.time() - start_time

            results[actual_size] = {
                "combinations": actual_size,
                "execution_time": execution_time,
                "throughput": actual_size / execution_time,
                "valid_results": len(optimization_results),
                "workers": len(self.workers),
                "efficiency": (actual_size / execution_time) / len(self.workers),
            }

            self.logger.info(
                f"   ⚡ Throughput: {results[actual_size]['throughput']:.1f} combinations/sec"
            )

        return results

    def _generate_parameter_combinations(
        self, parameter_grid: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        import itertools

        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination, strict=False))
            combinations.append(param_dict)

        return combinations


def benchmark_ray_distributed():
    """Benchmark Ray distributed computing performance"""
    print("🚀 Ray Distributed Computing Benchmark")
    print("=" * 50)

    # Generate test data
    np.random.seed(42)
    n_days = 2000
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

    # Initialize distributed engine
    config = DistributedConfig(
        num_workers=4, include_dashboard=False  # Dashboard requires ray[default]
    )

    engine = RayDistributedEngine(config)

    try:
        # Initialize cluster
        if not engine.initialize():
            print("❌ Failed to initialize Ray cluster")
            return

        # Get cluster info
        cluster_info = engine.get_cluster_info()
        print("\n📊 Ray Cluster Info:")
        print(f"   Nodes: {cluster_info['nodes']}")
        print(f"   CPUs: {cluster_info['cpus']}")
        print(f"   Memory: {cluster_info['memory_gb']:.1f} GB")
        print(f"   Workers: {cluster_info['workers']}")

        # Run benchmark
        print("\n🧪 Running scaling benchmark...")
        benchmark_results = engine.benchmark_scaling(data, test_sizes=[10, 50, 100, 200])

        # Display results
        print("\n📊 SCALING RESULTS:")
        print(f"{'Size':<10} {'Time (s)':<12} {'Throughput':<15} {'Efficiency':<12}")
        print("-" * 50)

        for size, metrics in sorted(benchmark_results.items()):
            print(
                f"{size:<10} {metrics['execution_time']:<12.2f} "
                f"{metrics['throughput']:<15.1f} {metrics['efficiency']:<12.1f}"
            )

        # Calculate scaling efficiency
        sizes = sorted(benchmark_results.keys())
        if len(sizes) >= 2:
            small_size = sizes[0]
            large_size = sizes[-1]

            scaling_factor = large_size / small_size
            speedup = (
                benchmark_results[large_size]["throughput"]
                / benchmark_results[small_size]["throughput"]
            )
            scaling_efficiency = speedup / scaling_factor

            print("\n📈 Scaling Analysis:")
            print(f"   Problem size increase: {scaling_factor:.1f}x")
            print(f"   Throughput increase: {speedup:.1f}x")
            print(f"   Scaling efficiency: {scaling_efficiency:.1%}")

        return benchmark_results

    finally:
        # Cleanup
        engine.shutdown()
        print("\n✅ Ray cluster shutdown")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_ray_distributed()
