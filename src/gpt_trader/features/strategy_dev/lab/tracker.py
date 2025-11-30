"""Experiment tracking and management.

Provides:
- ExperimentTracker: Manage experiment lifecycle and persistence
- Query and analysis of experiment results
- Export and comparison tools
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_trader.features.strategy_dev.lab.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
)
from gpt_trader.features.strategy_dev.lab.parameter_grid import ParameterGrid

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTracker:
    """Track and manage strategy experiments.

    Provides:
    - Experiment creation and lifecycle management
    - Persistence to JSON files
    - Query and filtering
    - Comparison and analysis
    """

    storage_path: Path | None = None
    experiments: dict[str, Experiment] = field(default_factory=dict)
    _callbacks: dict[str, list[Callable]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize storage and load existing experiments."""
        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_experiments()

    def _load_experiments(self) -> None:
        """Load experiments from storage."""
        if not self.storage_path:
            return

        experiments_file = self.storage_path / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file) as f:
                    data = json.load(f)

                for exp_data in data.get("experiments", []):
                    experiment = Experiment.from_dict(exp_data)
                    self.experiments[experiment.experiment_id] = experiment

                logger.info(f"Loaded {len(self.experiments)} experiments from storage")
            except Exception as e:
                logger.error(f"Error loading experiments: {e}")

    def _save_experiments(self) -> None:
        """Save experiments to storage."""
        if not self.storage_path:
            return

        experiments_file = self.storage_path / "experiments.json"
        data = {
            "last_updated": datetime.now().isoformat(),
            "experiments": [exp.to_dict() for exp in self.experiments.values()],
        }

        with open(experiments_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for experiment events.

        Events:
        - created: Experiment created
        - started: Experiment started running
        - completed: Experiment completed
        - failed: Experiment failed
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, experiment: Experiment) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(experiment)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")

    def create_experiment(
        self,
        name: str,
        strategy_name: str,
        parameters: dict[str, Any],
        description: str = "",
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        baseline_parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            strategy_name: Strategy being tested
            parameters: Parameters to test
            description: Description of experiment
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            baseline_parameters: Parameters for baseline comparison
            tags: Tags for organization

        Returns:
            Created experiment
        """
        experiment = Experiment(
            name=name,
            description=description,
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            baseline_parameters=baseline_parameters,
            tags=tags or [],
        )

        self.experiments[experiment.experiment_id] = experiment
        self._save_experiments()
        self._emit("created", experiment)

        logger.info(f"Created experiment: {experiment.experiment_id} - {name}")
        return experiment

    def create_grid_experiments(
        self,
        name_prefix: str,
        strategy_name: str,
        parameter_grid: ParameterGrid,
        description: str = "",
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        baseline_parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        max_experiments: int | None = None,
    ) -> list[Experiment]:
        """Create experiments from a parameter grid.

        Args:
            name_prefix: Prefix for experiment names
            strategy_name: Strategy being tested
            parameter_grid: Grid of parameters to test
            description: Base description
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            baseline_parameters: Parameters for baseline comparison
            tags: Tags for organization
            max_experiments: Maximum number to create (random sample if exceeded)

        Returns:
            List of created experiments
        """
        experiments = []

        # Get parameter combinations
        param_combinations = list(parameter_grid)
        if max_experiments and len(param_combinations) > max_experiments:
            param_combinations = parameter_grid.sample(max_experiments)
            logger.info(f"Sampled {max_experiments} from {len(parameter_grid)} combinations")

        for i, params in enumerate(param_combinations):
            name = f"{name_prefix}_{i + 1}"
            exp_description = f"{description}\nParameters: {params}"

            experiment = self.create_experiment(
                name=name,
                strategy_name=strategy_name,
                parameters=params,
                description=exp_description,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                baseline_parameters=baseline_parameters,
                tags=(tags or []) + ["grid_search"],
            )
            experiments.append(experiment)

        return experiments

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Mark experiment as started."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise KeyError(f"Experiment {experiment_id} not found")

        experiment.start()
        self._save_experiments()
        self._emit("started", experiment)
        return experiment

    def complete_experiment(self, experiment_id: str, result: ExperimentResult) -> Experiment:
        """Mark experiment as completed with result."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise KeyError(f"Experiment {experiment_id} not found")

        experiment.complete(result)
        self._save_experiments()
        self._emit("completed", experiment)

        logger.info(
            f"Completed experiment {experiment_id}: "
            f"return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.2f}"
        )
        return experiment

    def fail_experiment(self, experiment_id: str, error_message: str) -> Experiment:
        """Mark experiment as failed."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise KeyError(f"Experiment {experiment_id} not found")

        experiment.fail(error_message)
        self._save_experiments()
        self._emit("failed", experiment)

        logger.error(f"Experiment {experiment_id} failed: {error_message}")
        return experiment

    def query(
        self,
        status: ExperimentStatus | None = None,
        strategy_name: str | None = None,
        tags: list[str] | None = None,
        min_return: float | None = None,
        min_sharpe: float | None = None,
        since: datetime | None = None,
    ) -> list[Experiment]:
        """Query experiments with filters.

        Args:
            status: Filter by status
            strategy_name: Filter by strategy
            tags: Filter by tags (any match)
            min_return: Minimum total return
            min_sharpe: Minimum Sharpe ratio
            since: Created after this date

        Returns:
            List of matching experiments
        """
        results = []

        for experiment in self.experiments.values():
            # Status filter
            if status and experiment.status != status:
                continue

            # Strategy filter
            if strategy_name and experiment.strategy_name != strategy_name:
                continue

            # Tags filter
            if tags and not any(tag in experiment.tags for tag in tags):
                continue

            # Date filter
            if since and experiment.created_at < since:
                continue

            # Performance filters (only for completed)
            if experiment.status == ExperimentStatus.COMPLETED and experiment.result:
                if min_return is not None and experiment.result.total_return < min_return:
                    continue
                if min_sharpe is not None and experiment.result.sharpe_ratio < min_sharpe:
                    continue

            results.append(experiment)

        return results

    def get_pending(self) -> list[Experiment]:
        """Get all pending experiments."""
        return self.query(status=ExperimentStatus.PENDING)

    def get_completed(self) -> list[Experiment]:
        """Get all completed experiments."""
        return self.query(status=ExperimentStatus.COMPLETED)

    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        count: int = 10,
        strategy_name: str | None = None,
    ) -> list[Experiment]:
        """Get top performing experiments.

        Args:
            metric: Metric to rank by (sharpe_ratio, total_return, win_rate)
            count: Number of results
            strategy_name: Filter by strategy

        Returns:
            List of top experiments
        """
        completed = self.query(status=ExperimentStatus.COMPLETED, strategy_name=strategy_name)

        def get_metric(exp: Experiment) -> float:
            if not exp.result:
                return float("-inf")
            return getattr(exp.result, metric, float("-inf"))

        sorted_experiments = sorted(completed, key=get_metric, reverse=True)
        return sorted_experiments[:count]

    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
        """Compare multiple experiments.

        Args:
            experiment_ids: IDs of experiments to compare

        Returns:
            Comparison data
        """
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [e for e in experiments if e and e.result]

        if len(experiments) < 2:
            return {"error": "Need at least 2 completed experiments to compare"}

        comparison = {
            "experiments": [],
            "metrics_summary": {},
            "parameter_comparison": {},
        }

        # Collect metrics
        metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]

        for metric in metrics:
            values = [getattr(e.result, metric, 0) for e in experiments]
            comparison["metrics_summary"][metric] = {
                "values": values,
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "best_experiment_id": (
                    experiments[values.index(max(values))].experiment_id if values else None
                ),
            }

        # Compare parameters
        param_keys = set()
        for exp in experiments:
            param_keys.update(exp.parameters.keys())

        for key in param_keys:
            values = [exp.parameters.get(key) for exp in experiments]
            comparison["parameter_comparison"][key] = values

        # Add experiment summaries
        for exp in experiments:
            comparison["experiments"].append(
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "parameters": exp.parameters,
                    "metrics": exp.result.to_dict()["metrics"] if exp.result else {},
                }
            )

        return comparison

    def get_parameter_impact(self, strategy_name: str, parameter_name: str) -> dict[str, Any]:
        """Analyze impact of a parameter across experiments.

        Args:
            strategy_name: Strategy to analyze
            parameter_name: Parameter to analyze

        Returns:
            Parameter impact analysis
        """
        completed = self.query(status=ExperimentStatus.COMPLETED, strategy_name=strategy_name)

        # Group by parameter value
        value_results: dict[Any, list[Experiment]] = {}
        for exp in completed:
            value = exp.parameters.get(parameter_name)
            if value is not None:
                if value not in value_results:
                    value_results[value] = []
                value_results[value].append(exp)

        # Calculate metrics per value
        analysis = {
            "parameter": parameter_name,
            "strategy": strategy_name,
            "value_analysis": {},
        }

        for value, experiments in value_results.items():
            returns = [e.result.total_return for e in experiments if e.result]
            sharpes = [e.result.sharpe_ratio for e in experiments if e.result]

            analysis["value_analysis"][str(value)] = {
                "count": len(experiments),
                "avg_return": sum(returns) / len(returns) if returns else 0,
                "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0,
                "best_return": max(returns) if returns else 0,
                "best_sharpe": max(sharpes) if sharpes else 0,
            }

        return analysis

    def export_results(self, output_path: Path, format: str = "json") -> None:
        """Export all experiment results.

        Args:
            output_path: Path to export file
            format: Export format (json, csv)
        """
        output_path = Path(output_path)

        if format == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "total_experiments": len(self.experiments),
                "experiments": [exp.to_dict() for exp in self.experiments.values()],
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "id",
                        "name",
                        "strategy",
                        "status",
                        "total_return",
                        "sharpe_ratio",
                        "max_drawdown",
                        "win_rate",
                        "total_trades",
                        "created_at",
                    ]
                )

                # Rows
                for exp in self.experiments.values():
                    result = exp.result
                    writer.writerow(
                        [
                            exp.experiment_id,
                            exp.name,
                            exp.strategy_name,
                            exp.status.value,
                            result.total_return if result else "",
                            result.sharpe_ratio if result else "",
                            result.max_drawdown if result else "",
                            result.win_rate if result else "",
                            result.total_trades if result else "",
                            exp.created_at.isoformat(),
                        ]
                    )

        logger.info(f"Exported {len(self.experiments)} experiments to {output_path}")

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of all experiments."""
        by_status = {}
        for status in ExperimentStatus:
            by_status[status.value] = len(
                [e for e in self.experiments.values() if e.status == status]
            )

        by_strategy = {}
        for exp in self.experiments.values():
            if exp.strategy_name not in by_strategy:
                by_strategy[exp.strategy_name] = 0
            by_strategy[exp.strategy_name] += 1

        # Best performers
        best = self.get_best_experiments(count=5)
        best_summary = [
            {
                "id": e.experiment_id,
                "name": e.name,
                "sharpe": e.result.sharpe_ratio if e.result else 0,
                "return": e.result.total_return if e.result else 0,
            }
            for e in best
        ]

        return {
            "total_experiments": len(self.experiments),
            "by_status": by_status,
            "by_strategy": by_strategy,
            "best_performers": best_summary,
        }

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: ID of experiment to delete

        Returns:
            True if deleted, False if not found
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self._save_experiments()
            return True
        return False

    def clear_all(self, confirm: bool = False) -> int:
        """Clear all experiments.

        Args:
            confirm: Must be True to actually clear

        Returns:
            Number of experiments cleared
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear all experiments")

        count = len(self.experiments)
        self.experiments.clear()
        self._save_experiments()
        return count
