"""
Computational Efficiency Analyzer
Phase 2.5 - Day 9

Analyzes and optimizes computational efficiency of ML models.
"""

import cProfile
import io
import logging
import pstats
import time
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Computational efficiency metrics"""

    # Time metrics
    total_time: float  # seconds
    training_time: float
    inference_time: float
    feature_engineering_time: float

    # Memory metrics
    peak_memory_mb: float
    average_memory_mb: float
    memory_per_sample_kb: float

    # CPU metrics
    cpu_usage_percent: float
    n_cores_used: int
    parallelization_efficiency: float

    # Throughput metrics
    samples_per_second: float
    predictions_per_second: float

    # Scalability metrics
    time_complexity: str  # O(n), O(n²), etc.
    space_complexity: str
    scalability_score: float  # 0-1, higher is better

    # Optimization potential
    bottlenecks: list[str]
    optimization_suggestions: list[str]
    potential_speedup: float  # Estimated speedup factor


@dataclass
class ScalabilityTest:
    """Scalability test configuration and results"""

    sample_sizes: list[int]
    feature_counts: list[int]

    # Results
    time_by_samples: dict[int, float]
    memory_by_samples: dict[int, float]
    time_by_features: dict[int, float]
    memory_by_features: dict[int, float]

    # Complexity estimates
    estimated_time_complexity: str
    estimated_space_complexity: str


class EfficiencyAnalyzer:
    """
    Analyzes computational efficiency of ML models.

    Features:
    - Time and memory profiling
    - Scalability analysis
    - Bottleneck detection
    - Optimization recommendations
    - Parallel processing analysis
    """

    def __init__(self):
        """Initialize analyzer"""
        self.profiling_enabled = False
        self.memory_snapshots = []
        self.cpu_samples = []

        logger.info("EfficiencyAnalyzer initialized")

    def analyze_model_efficiency(
        self, model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
    ) -> EfficiencyMetrics:
        """
        Analyze computational efficiency of a model.

        Args:
            model: Model to analyze
            X_train: Training features
            y_train: Training target
            X_test: Test features

        Returns:
            Efficiency metrics
        """
        logger.info(f"Analyzing efficiency for {type(model).__name__}")

        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()

        # Feature engineering time (if applicable)
        fe_time = self._measure_feature_engineering_time(X_train)

        # Training efficiency
        train_metrics = self._measure_training_efficiency(model, X_train, y_train)

        # Inference efficiency
        inference_metrics = self._measure_inference_efficiency(model, X_test)

        # Memory analysis
        memory_metrics = self._analyze_memory_usage(model, X_train, X_test)

        # CPU analysis
        cpu_metrics = self._analyze_cpu_usage(model, X_train, y_train)

        # Scalability analysis
        scalability = self._analyze_scalability(model, X_train, y_train)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(train_metrics, inference_metrics, memory_metrics)

        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(bottlenecks, model, scalability)

        # Stop monitoring
        tracemalloc.stop()

        return EfficiencyMetrics(
            total_time=train_metrics["time"] + inference_metrics["time"],
            training_time=train_metrics["time"],
            inference_time=inference_metrics["time"],
            feature_engineering_time=fe_time,
            peak_memory_mb=memory_metrics["peak_mb"],
            average_memory_mb=memory_metrics["average_mb"],
            memory_per_sample_kb=memory_metrics["per_sample_kb"],
            cpu_usage_percent=cpu_metrics["usage_percent"],
            n_cores_used=cpu_metrics["n_cores"],
            parallelization_efficiency=cpu_metrics["parallel_efficiency"],
            samples_per_second=train_metrics["throughput"],
            predictions_per_second=inference_metrics["throughput"],
            time_complexity=scalability["time_complexity"],
            space_complexity=scalability["space_complexity"],
            scalability_score=scalability["score"],
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            potential_speedup=self._estimate_speedup(bottlenecks, suggestions),
        )

    def _measure_feature_engineering_time(self, X: pd.DataFrame) -> float:
        """Measure time for feature engineering"""
        start = time.time()

        # Simulate feature engineering operations
        _ = X.rolling(window=20).mean()
        _ = X.pct_change()
        _ = X.shift(1)

        return time.time() - start

    def _measure_training_efficiency(self, model, X, y) -> dict:
        """Measure training efficiency"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Profile training
        if self.profiling_enabled:
            profiler = cProfile.Profile()
            profiler.enable()

        model.fit(X, y)

        if self.profiling_enabled:
            profiler.disable()
            # Analyze profile
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(10)
            profile_output = s.getvalue()
        else:
            profile_output = ""

        training_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory

        return {
            "time": training_time,
            "memory_mb": memory_used,
            "throughput": len(X) / training_time,
            "profile": profile_output,
        }

    def _measure_inference_efficiency(self, model, X) -> dict:
        """Measure inference efficiency"""
        # Warm up
        _ = model.predict(X[:10])

        # Measure batch inference
        start_time = time.time()
        predictions = model.predict(X)
        batch_time = time.time() - start_time

        # Measure single sample inference
        single_times = []
        for i in range(min(100, len(X))):
            start = time.time()
            _ = model.predict(X.iloc[i : i + 1])
            single_times.append(time.time() - start)

        avg_single_time = np.mean(single_times)

        return {
            "time": batch_time,
            "single_time": avg_single_time,
            "throughput": len(X) / batch_time,
            "batch_speedup": (avg_single_time * len(X)) / batch_time,
        }

    def _analyze_memory_usage(self, model, X_train, X_test) -> dict:
        """Analyze memory usage patterns"""
        # Get model size
        model_size = self._get_model_size(model)

        # Track memory during operations
        memory_samples = []
        process = psutil.Process()

        for _ in range(10):
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
            time.sleep(0.1)

        # Calculate metrics
        peak_memory = max(memory_samples)
        avg_memory = np.mean(memory_samples)

        # Memory per sample
        data_memory = (
            (X_train.memory_usage(deep=True).sum() + X_test.memory_usage(deep=True).sum())
            / 1024
            / 1024
        )
        memory_per_sample = (data_memory / (len(X_train) + len(X_test))) * 1024  # KB

        return {
            "peak_mb": peak_memory,
            "average_mb": avg_memory,
            "model_size_mb": model_size,
            "per_sample_kb": memory_per_sample,
        }

    def _analyze_cpu_usage(self, model, X, y) -> dict:
        """Analyze CPU usage patterns"""
        cpu_samples = []
        process = psutil.Process()

        # Monitor CPU during training
        def monitor_cpu():
            while self.monitoring:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)

        self.monitoring = True
        from threading import Thread

        monitor_thread = Thread(target=monitor_cpu)
        monitor_thread.start()

        # Train model
        model.fit(X, y)

        self.monitoring = False
        monitor_thread.join()

        # Analyze CPU usage
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        n_cores = psutil.cpu_count()

        # Estimate parallelization efficiency
        if hasattr(model, "n_jobs"):
            if model.n_jobs == -1 or model.n_jobs == n_cores:
                parallel_efficiency = avg_cpu / (100 * n_cores)
            else:
                parallel_efficiency = avg_cpu / (100 * model.n_jobs)
        else:
            parallel_efficiency = avg_cpu / 100

        return {
            "usage_percent": avg_cpu,
            "n_cores": n_cores,
            "parallel_efficiency": parallel_efficiency,
        }

    def _analyze_scalability(self, model, X, y) -> dict:
        """Analyze model scalability"""
        sample_sizes = [100, 500, 1000, min(5000, len(X))]
        times = []
        memories = []

        for n in sample_sizes:
            if n > len(X):
                break

            X_subset = X.iloc[:n]
            y_subset = y.iloc[:n]

            # Measure time
            start = time.time()
            model.fit(X_subset, y_subset)
            times.append(time.time() - start)

            # Measure memory
            memories.append(psutil.Process().memory_info().rss / 1024 / 1024)

        # Estimate complexity
        time_complexity = self._estimate_complexity(sample_sizes[: len(times)], times)
        space_complexity = self._estimate_complexity(sample_sizes[: len(memories)], memories)

        # Calculate scalability score
        # Good scalability: linear or n log n
        # Poor scalability: quadratic or worse
        if "n²" in time_complexity or "n³" in time_complexity:
            score = 0.3
        elif "n log n" in time_complexity:
            score = 0.8
        elif "n" in time_complexity:
            score = 0.9
        else:
            score = 0.5

        return {
            "time_complexity": time_complexity,
            "space_complexity": space_complexity,
            "score": score,
            "sample_sizes": sample_sizes[: len(times)],
            "times": times,
            "memories": memories,
        }

    def _estimate_complexity(self, sizes: list[int], values: list[float]) -> str:
        """Estimate computational complexity from data"""
        if len(sizes) < 2:
            return "O(?)"

        # Fit different complexity models
        log_sizes = np.log(sizes)
        log_values = np.log(values)

        # Linear regression in log-log space
        slope = np.polyfit(log_sizes, log_values, 1)[0]

        # Classify complexity based on slope
        if slope < 1.2:
            return "O(n)"
        elif slope < 1.8:
            return "O(n log n)"
        elif slope < 2.2:
            return "O(n²)"
        elif slope < 3.2:
            return "O(n³)"
        else:
            return f"O(n^{slope:.1f})"

    def _identify_bottlenecks(self, train_metrics, inference_metrics, memory_metrics) -> list[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Time bottlenecks
        if train_metrics["time"] > 10:
            bottlenecks.append("slow_training")
        if inference_metrics["single_time"] > 0.1:
            bottlenecks.append("slow_inference")

        # Memory bottlenecks
        if memory_metrics["peak_mb"] > 1000:
            bottlenecks.append("high_memory_usage")
        if memory_metrics["model_size_mb"] > 100:
            bottlenecks.append("large_model_size")

        # Efficiency bottlenecks
        if inference_metrics.get("batch_speedup", 1) < 5:
            bottlenecks.append("poor_batch_efficiency")

        return bottlenecks

    def _generate_optimization_suggestions(self, bottlenecks, model, scalability) -> list[str]:
        """Generate optimization suggestions"""
        suggestions = []

        if "slow_training" in bottlenecks:
            suggestions.append("Use early stopping to reduce training time")
            suggestions.append("Reduce model complexity (fewer trees/layers)")
            if hasattr(model, "n_jobs"):
                suggestions.append("Increase n_jobs for parallel processing")

        if "slow_inference" in bottlenecks:
            suggestions.append("Use model quantization or pruning")
            suggestions.append("Implement batch prediction")
            suggestions.append("Consider using ONNX for inference")

        if "high_memory_usage" in bottlenecks:
            suggestions.append("Use sparse matrices for features")
            suggestions.append("Implement incremental learning")
            suggestions.append("Reduce feature dimensionality")

        if "large_model_size" in bottlenecks:
            suggestions.append("Use model compression techniques")
            suggestions.append("Reduce ensemble size")
            suggestions.append("Use feature selection to reduce inputs")

        if scalability["score"] < 0.5:
            suggestions.append("Consider using SGD-based models for better scalability")
            suggestions.append("Implement mini-batch processing")

        return suggestions

    def _estimate_speedup(self, bottlenecks, suggestions) -> float:
        """Estimate potential speedup from optimizations"""
        speedup = 1.0

        if "slow_training" in bottlenecks and "Increase n_jobs" in suggestions:
            speedup *= 2.0  # Parallel processing

        if "slow_inference" in bottlenecks and "batch prediction" in suggestions:
            speedup *= 5.0  # Batch efficiency

        if "high_memory_usage" in bottlenecks and "sparse matrices" in suggestions:
            speedup *= 1.5  # Memory efficiency

        return speedup

    def _get_model_size(self, model) -> float:
        """Get model size in MB using secure serialization"""
        try:
            # Use joblib for sklearn models or estimate based on parameters
            import os
            import tempfile

            import joblib

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump(model, tmp.name)
                size_mb = os.path.getsize(tmp.name) / 1024 / 1024
                os.unlink(tmp.name)
                return size_mb
        except:
            # Fallback: estimate based on model parameters if available
            try:
                if hasattr(model, "coef_"):
                    # Linear models
                    return len(model.coef_) * 8 / 1024 / 1024  # 8 bytes per float64
                elif hasattr(model, "estimators_"):
                    # Ensemble models
                    return len(model.estimators_) * 100 / 1024 / 1024  # Rough estimate
                else:
                    return 1.0  # Default 1MB estimate
            except:
                return 1.0

    def benchmark_parallel_processing(
        self, func: Callable, data: list, n_workers_list: list[int] = [1, 2, 4, 8]
    ) -> dict:
        """Benchmark parallel processing efficiency"""
        results = {}

        for n_workers in n_workers_list:
            start = time.time()

            if n_workers == 1:
                # Sequential processing
                results_seq = [func(d) for d in data]
            else:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results_par = list(executor.map(func, data))

            elapsed = time.time() - start

            results[n_workers] = {
                "time": elapsed,
                "throughput": len(data) / elapsed,
                "speedup": results[1]["time"] / elapsed if 1 in results else 1,
            }

        return results

    def plot_efficiency_analysis(
        self, metrics: EfficiencyMetrics, scalability_data: dict | None = None
    ):
        """Plot efficiency analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Time breakdown
        ax = axes[0, 0]
        times = [metrics.feature_engineering_time, metrics.training_time, metrics.inference_time]
        labels = ["Feature Eng.", "Training", "Inference"]
        ax.pie(times, labels=labels, autopct="%1.1f%%")
        ax.set_title("Time Breakdown")

        # 2. Memory usage
        ax = axes[0, 1]
        memory_data = {
            "Peak": metrics.peak_memory_mb,
            "Average": metrics.average_memory_mb,
            "Per Sample (KB)": metrics.memory_per_sample_kb,
        }
        ax.bar(range(len(memory_data)), list(memory_data.values()))
        ax.set_xticks(range(len(memory_data)))
        ax.set_xticklabels(list(memory_data.keys()))
        ax.set_ylabel("Memory (MB/KB)")
        ax.set_title("Memory Usage")

        # 3. Scalability
        if scalability_data:
            ax = axes[1, 0]
            if "sample_sizes" in scalability_data:
                ax.plot(
                    scalability_data["sample_sizes"], scalability_data["times"], "o-", label="Time"
                )
                ax.set_xlabel("Sample Size")
                ax.set_ylabel("Time (s)")
                ax.set_title(f"Scalability: {metrics.time_complexity}")
                ax.legend()

        # 4. Optimization potential
        ax = axes[1, 1]
        ax.axis("off")

        opt_text = f"""
Optimization Analysis:

Bottlenecks:
{chr(10).join('• ' + b for b in metrics.bottlenecks[:5])}

Suggestions:
{chr(10).join('• ' + s for s in metrics.optimization_suggestions[:5])}

Potential Speedup: {metrics.potential_speedup:.1f}x
Scalability Score: {metrics.scalability_score:.2f}
        """

        ax.text(0.1, 0.5, opt_text, fontsize=10, verticalalignment="center", family="monospace")
        ax.set_title("Optimization Potential")

        plt.suptitle("Efficiency Analysis", fontsize=14)
        plt.tight_layout()

        return fig


def create_efficiency_analyzer() -> EfficiencyAnalyzer:
    """Create efficiency analyzer instance"""
    return EfficiencyAnalyzer()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # Generate sample data
    n_samples = 5000
    n_features = 100

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_samples))

    X_test = pd.DataFrame(
        np.random.randn(1000, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create model
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Analyze efficiency
    analyzer = create_efficiency_analyzer()
    metrics = analyzer.analyze_model_efficiency(model, X_train, y_train, X_test)

    # Display results
    print("\nEfficiency Analysis Results:")
    print("=" * 50)
    print(f"Total Time: {metrics.total_time:.2f}s")
    print(f"Training Time: {metrics.training_time:.2f}s")
    print(f"Inference Time: {metrics.inference_time:.3f}s")
    print(f"Peak Memory: {metrics.peak_memory_mb:.1f} MB")
    print(f"Samples/sec: {metrics.samples_per_second:.1f}")
    print(f"Predictions/sec: {metrics.predictions_per_second:.1f}")
    print(f"Time Complexity: {metrics.time_complexity}")
    print(f"Scalability Score: {metrics.scalability_score:.2f}")

    if metrics.bottlenecks:
        print(f"\nBottlenecks: {', '.join(metrics.bottlenecks)}")

    if metrics.optimization_suggestions:
        print("\nOptimization Suggestions:")
        for suggestion in metrics.optimization_suggestions[:3]:
            print(f"  • {suggestion}")

    print(f"\nPotential Speedup: {metrics.potential_speedup:.1f}x")
