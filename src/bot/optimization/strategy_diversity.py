"""
Strategy diversity tracking and analysis for evolutionary optimization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StrategyDiversityTracker:
    """Tracks and analyzes diverse strategies discovered during optimization."""

    def __init__(self, output_dir: Path, strategy_config: Any) -> None:
        self.output_dir = output_dir
        self.strategy_config = strategy_config
        self.diverse_strategies: list[dict[str, Any]] = []
        self.strategy_clusters: list[dict[str, Any]] = []
        self.performance_thresholds = {
            "min_sharpe": 0.0,
            "min_cagr": 0.0,
            "max_drawdown": 0.5,
            "min_trades": 5,
        }

        # Create diversity analysis directory
        self.diversity_dir = output_dir / "diversity_analysis"
        # Ensure full directory path exists
        self.diversity_dir.mkdir(parents=True, exist_ok=True)

    def add_strategy(self, params: dict[str, Any], metrics: dict[str, Any]) -> bool:
        """
        Add a strategy if it meets diversity and performance criteria.

        Returns:
            True if strategy was added, False otherwise
        """
        # Check performance thresholds
        if not self._meets_performance_thresholds(metrics):
            return False

        # Check if strategy is diverse enough
        if self._is_diverse_strategy(params, metrics):
            strategy_data = {
                "params": params,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "diversity_score": self._calculate_diversity_score(params, metrics),
            }
            self.diverse_strategies.append(strategy_data)
            logger.info(f"Added diverse strategy with Sharpe: {metrics.get('sharpe', 0):.4f}")
            return True

        return False

    def _meets_performance_thresholds(self, metrics: dict[str, Any]) -> bool:
        """Check if strategy meets minimum performance criteria."""
        sharpe = metrics.get("sharpe", float("-inf"))
        cagr = metrics.get("cagr", float("-inf"))
        max_dd = metrics.get("max_drawdown", float("inf"))
        trades = metrics.get("n_trades", 0)

        return (
            sharpe >= self.performance_thresholds["min_sharpe"]
            and cagr >= self.performance_thresholds["min_cagr"]
            and max_dd <= self.performance_thresholds["max_drawdown"]
            and trades >= self.performance_thresholds["min_trades"]
        )

    def _is_diverse_strategy(self, params: dict[str, Any], metrics: dict[str, Any]) -> bool:
        """Check if strategy is diverse from existing ones."""
        if not self.diverse_strategies:
            return True

        # Calculate parameter distance from existing strategies
        min_distance = float("inf")
        for existing in self.diverse_strategies:
            distance = self._calculate_parameter_distance(params, existing["params"])
            min_distance = min(min_distance, distance)

        # Strategy is diverse if it's sufficiently different
        return min_distance > 0.3  # Threshold for diversity

    def _calculate_parameter_distance(
        self, params1: dict[str, Any], params2: dict[str, Any]
    ) -> float:
        """Calculate normalized distance between parameter sets."""
        if not params1 or not params2:
            return 0.0

        distances = []
        for key in set(params1.keys()) & set(params2.keys()):
            val1 = params1[key]
            val2 = params2[key]

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Normalize by parameter range
                param_def = self.strategy_config.parameters.get(key)
                if (
                    param_def
                    and param_def.min_value is not None
                    and param_def.max_value is not None
                ):
                    range_size = param_def.max_value - param_def.min_value
                    if range_size > 0:
                        normalized_dist = abs(val1 - val2) / range_size
                        distances.append(normalized_dist)
                    else:
                        distances.append(0.0)
                else:
                    # Fallback: use absolute difference
                    distances.append(abs(val1 - val2))
            elif isinstance(val1, bool) and isinstance(val2, bool):
                # Boolean parameters: 1 if different, 0 if same
                distances.append(1.0 if val1 != val2 else 0.0)
            else:
                # String or other types: 1 if different, 0 if same
                distances.append(1.0 if val1 != val2 else 0.0)

        return np.mean(distances) if distances else 0.0

    def _calculate_diversity_score(self, params: dict[str, Any], metrics: dict[str, Any]) -> float:
        """Calculate overall diversity score for a strategy."""
        # Combine parameter diversity and performance uniqueness
        param_diversity = self._calculate_parameter_diversity(params)
        performance_uniqueness = self._calculate_performance_uniqueness(metrics)

        # Weighted combination
        return 0.7 * param_diversity + 0.3 * performance_uniqueness

    def _calculate_parameter_diversity(self, params: dict[str, Any]) -> float:
        """Calculate how unique the parameter combination is."""
        if not self.diverse_strategies:
            return 1.0

        # Calculate average distance from existing strategies
        distances = []
        for existing in self.diverse_strategies:
            distance = self._calculate_parameter_distance(params, existing["params"])
            distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _calculate_performance_uniqueness(self, metrics: dict[str, Any]) -> float:
        """Calculate how unique the performance profile is."""
        if not self.diverse_strategies:
            return 1.0

        # Calculate performance distance from existing strategies
        distances = []
        for existing in self.diverse_strategies:
            existing_metrics = existing["metrics"]

            # Normalize metrics for comparison
            sharpe_diff = abs(metrics.get("sharpe", 0) - existing_metrics.get("sharpe", 0))
            cagr_diff = abs(metrics.get("cagr", 0) - existing_metrics.get("cagr", 0))
            dd_diff = abs(metrics.get("max_drawdown", 0) - existing_metrics.get("max_drawdown", 0))

            # Combine differences (normalize by typical ranges)
            performance_distance = (sharpe_diff / 2.0 + cagr_diff / 0.5 + dd_diff / 0.3) / 3.0
            distances.append(performance_distance)

        return np.mean(distances) if distances else 0.0

    def cluster_strategies(self, n_clusters: int = 5) -> list[dict[str, Any]]:
        """Cluster strategies to identify distinct strategy types."""
        if len(self.diverse_strategies) < n_clusters:
            logger.warning(f"Not enough diverse strategies for {n_clusters} clusters")
            return []

        # Extract parameter vectors
        param_vectors = []
        for strategy in self.diverse_strategies:
            vector = self._params_to_vector(strategy["params"])
            param_vectors.append(vector)

        # Normalize parameters
        scaler = StandardScaler()
        normalized_vectors = scaler.fit_transform(param_vectors)

        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(self.diverse_strategies)), random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_vectors)

        # Group strategies by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.diverse_strategies[i])

        # Analyze each cluster
        cluster_analysis = []
        for cluster_id, strategies in clusters.items():
            cluster_info = self._analyze_cluster(cluster_id, strategies)
            cluster_analysis.append(cluster_info)

        self.strategy_clusters = cluster_analysis
        return cluster_analysis

    def _params_to_vector(self, params: dict[str, Any]) -> list[float]:
        """Convert parameters to numerical vector for clustering."""
        vector = []
        for param_name, param_def in self.strategy_config.parameters.items():
            value = params.get(param_name, param_def.default)

            if isinstance(value, int | float):
                # Normalize by parameter range
                if param_def.min_value is not None and param_def.max_value is not None:
                    range_size = param_def.max_value - param_def.min_value
                    if range_size > 0:
                        normalized = (value - param_def.min_value) / range_size
                    else:
                        normalized = 0.0
                else:
                    normalized = float(value)
                vector.append(normalized)
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(0.0)  # Default for other types

        return vector

    def _analyze_cluster(self, cluster_id: int, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze a cluster of strategies."""
        if not strategies:
            return {}

        # Calculate cluster statistics
        sharpes = [s["metrics"].get("sharpe", 0) for s in strategies]
        cagrs = [s["metrics"].get("cagr", 0) for s in strategies]
        drawdowns = [s["metrics"].get("max_drawdown", 0) for s in strategies]

        # Find best strategy in cluster
        best_strategy = max(strategies, key=lambda x: x["metrics"].get("sharpe", 0))

        # Calculate parameter ranges in cluster
        param_ranges = {}
        for param_name in self.strategy_config.parameters.keys():
            values = [
                s["params"].get(param_name)
                for s in strategies
                if s["params"].get(param_name) is not None
            ]
            if values:
                param_ranges[param_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return {
            "cluster_id": cluster_id,
            "size": len(strategies),
            "best_strategy": best_strategy,
            "performance_stats": {
                "sharpe": {"mean": np.mean(sharpes), "std": np.std(sharpes), "max": max(sharpes)},
                "cagr": {"mean": np.mean(cagrs), "std": np.std(cagrs), "max": max(cagrs)},
                "drawdown": {
                    "mean": np.mean(drawdowns),
                    "std": np.std(drawdowns),
                    "min": min(drawdowns),
                },
            },
            "parameter_ranges": param_ranges,
            "strategies": strategies,
        }

    def save_diverse_strategies(self) -> None:
        """Save diverse strategies to files."""
        if not self.diverse_strategies:
            logger.info("No diverse strategies to save")
            return

        # Sort by diversity score
        sorted_strategies = sorted(
            self.diverse_strategies, key=lambda x: x["diversity_score"], reverse=True
        )

        # Save all diverse strategies
        strategies_file = self.diversity_dir / "diverse_strategies.json"
        with open(strategies_file, "w") as f:
            json.dump(sorted_strategies, f, indent=2, default=str)

        # Save top strategies
        top_strategies = sorted_strategies[:10]
        top_file = self.diversity_dir / "top_diverse_strategies.json"
        with open(top_file, "w") as f:
            json.dump(top_strategies, f, indent=2, default=str)

        # Create summary report
        self._create_diversity_report(sorted_strategies)

        logger.info(f"Saved {len(sorted_strategies)} diverse strategies to {self.diversity_dir}")

    def _create_diversity_report(self, strategies: list[dict[str, Any]]) -> None:
        """Create a comprehensive diversity analysis report."""
        if not strategies:
            return

        report = {
            "summary": {
                "total_diverse_strategies": len(strategies),
                "date_generated": datetime.now().isoformat(),
                "performance_range": {
                    "sharpe": {
                        "min": min(s["metrics"].get("sharpe", 0) for s in strategies),
                        "max": max(s["metrics"].get("sharpe", 0) for s in strategies),
                        "mean": np.mean([s["metrics"].get("sharpe", 0) for s in strategies]),
                    },
                    "cagr": {
                        "min": min(s["metrics"].get("cagr", 0) for s in strategies),
                        "max": max(s["metrics"].get("cagr", 0) for s in strategies),
                        "mean": np.mean([s["metrics"].get("cagr", 0) for s in strategies]),
                    },
                },
                "diversity_scores": {
                    "min": min(s["diversity_score"] for s in strategies),
                    "max": max(s["diversity_score"] for s in strategies),
                    "mean": np.mean([s["diversity_score"] for s in strategies]),
                },
            },
            "strategy_types": self._identify_strategy_types(strategies),
            "parameter_analysis": self._analyze_parameter_distributions(strategies),
        }

        # Save report
        report_file = self.diversity_dir / "diversity_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def _identify_strategy_types(self, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify different types of strategies based on parameter patterns."""
        strategy_types = {
            "conservative": [],
            "aggressive": [],
            "short_term": [],
            "long_term": [],
            "high_frequency": [],
            "momentum_based": [],
        }

        for strategy in strategies:
            params = strategy["params"]

            # Conservative strategies (low risk, high confirmation)
            if params.get("risk_pct", 0) < 0.5 and params.get("entry_confirm", 0) > 2:
                strategy_types["conservative"].append(strategy)

            # Aggressive strategies (high risk, low confirmation)
            if params.get("risk_pct", 0) > 2.0 and params.get("entry_confirm", 0) <= 1:
                strategy_types["aggressive"].append(strategy)

            # Short-term strategies
            if params.get("donchian_lookback", 0) < 50 and params.get("atr_period", 0) < 20:
                strategy_types["short_term"].append(strategy)

            # Long-term strategies
            if params.get("donchian_lookback", 0) > 200 and params.get("regime_window", 0) > 500:
                strategy_types["long_term"].append(strategy)

            # High-frequency strategies
            if params.get("cooldown", 0) == 0 and params.get("entry_confirm", 0) == 0:
                strategy_types["high_frequency"].append(strategy)

            # Momentum-based strategies
            if (
                params.get("momentum_lookback", 0) > 0
                and params.get("trend_strength_threshold", 0) > 0
            ):
                strategy_types["momentum_based"].append(strategy)

        # Convert to summary
        return {k: len(v) for k, v in strategy_types.items()}

    def _analyze_parameter_distributions(self, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze parameter distributions across diverse strategies."""
        analysis = {}

        for param_name, _param_def in self.strategy_config.parameters.items():
            values = [
                s["params"].get(param_name)
                for s in strategies
                if s["params"].get(param_name) is not None
            ]
            if values:
                analysis[param_name] = {
                    "count": len(values),
                    "unique_values": len(set(values)),
                    "distribution": {
                        "min": min(values),
                        "max": max(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                    },
                }

        return analysis

    def get_strategy_recommendations(self) -> list[dict[str, Any]]:
        """Get recommendations for different market conditions."""
        if not self.diverse_strategies:
            return []

        recommendations = []

        # Find best strategies for different scenarios
        scenarios = {
            "bull_market": lambda s: s["metrics"].get("cagr", 0) > 0.1
            and s["metrics"].get("sharpe", 0) > 0.5,
            "bear_market": lambda s: s["metrics"].get("max_drawdown", 0) < 0.2
            and s["metrics"].get("sharpe", 0) > 0.3,
            "volatile_market": lambda s: s["params"].get("atr_k", 0) > 3.0
            and s["metrics"].get("sharpe", 0) > 0.4,
            "trending_market": lambda s: s["params"].get("donchian_lookback", 0) > 100
            and s["metrics"].get("sharpe", 0) > 0.6,
            "sideways_market": lambda s: s["params"].get("entry_confirm", 0) > 2
            and s["metrics"].get("sharpe", 0) > 0.3,
        }

        for scenario, condition in scenarios.items():
            matching_strategies = [s for s in self.diverse_strategies if condition(s)]
            if matching_strategies:
                best_strategy = max(
                    matching_strategies, key=lambda x: x["metrics"].get("sharpe", 0)
                )
                recommendations.append(
                    {
                        "scenario": scenario,
                        "strategy": best_strategy,
                        "alternative_count": len(matching_strategies),
                    }
                )

        return recommendations
