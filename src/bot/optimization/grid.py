"""
Grid search optimization implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class GridOptimizer:
    """Grid search optimizer for systematic parameter exploration."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def optimize(
        self, evaluate_func: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Run grid search optimization.

        Args:
            evaluate_func: Function that evaluates a parameter combination
                          and returns results dictionary

        Returns:
            List of evaluation results
        """
        logger.info("Starting grid search optimization")

        # Get all parameter combinations
        combinations = self.config.parameter_space.get_grid_combinations()

        if not combinations:
            logger.warning("No parameter combinations defined for grid search")
            return []

        logger.info(f"Grid search will evaluate {len(combinations)} combinations")

        # Evaluate each combination with optional early pruning
        results = []
        prune_after = (
            int(self.config.early_stop_min) if hasattr(self.config, "early_stop_min") else 0
        )
        for i, params in enumerate(combinations):
            logger.debug(f"Evaluating combination {i+1}/{len(combinations)}: {params}")

            try:
                result = evaluate_func(params)
                result["params"] = params
                results.append(result)

                # Early pruning based on performance thresholds
                if prune_after > 0 and len(results) >= prune_after:
                    results = [
                        r
                        for r in results
                        if (
                            r.get("sharpe", -1e9) >= self.config.min_sharpe
                            and r.get("max_drawdown", 1e9) <= self.config.max_drawdown
                        )
                    ]

                # Log progress (less frequent for large grids)
                progress_interval = max(1, min(50, len(combinations) // 10))
                if (i + 1) % progress_interval == 0 or i == len(combinations) - 1:
                    logger.info(
                        f"Grid search progress: {i+1}/{len(combinations)} combinations evaluated"
                    )

            except Exception as e:
                logger.error(f"Failed to evaluate combination {params}: {e}")
                # Add error result
                results.append(
                    {
                        "params": params,
                        "sharpe": float("-inf"),
                        "cagr": float("-inf"),
                        "max_drawdown": float("inf"),
                        "total_return": float("-inf"),
                        "n_trades": 0,
                        "error": str(e),
                    }
                )

        logger.info(f"Grid search complete. Evaluated {len(results)} combinations")
        return results

    def get_parameter_ranges(self) -> dict[str, list[Any]]:
        """Get the parameter ranges used for grid search."""
        return self.config.parameter_space.grid_ranges

    def estimate_computation_time(self, avg_evaluation_time: float) -> float:
        """
        Estimate total computation time for grid search.

        Args:
            avg_evaluation_time: Average time per evaluation in seconds

        Returns:
            Estimated total time in seconds
        """
        combinations = self.config.parameter_space.get_grid_combinations()
        total_evaluations = len(combinations)

        # Account for parallelization
        effective_evaluations = total_evaluations / max(1, self.config.max_workers)

        return effective_evaluations * avg_evaluation_time
