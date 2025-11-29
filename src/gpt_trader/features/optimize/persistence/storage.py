"""Persistence for optimization runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_trader.features.optimize.runner.batch_runner import TrialResult
from gpt_trader.features.optimize.types import OptimizationConfig
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="optimize_storage")


@dataclass
class OptimizationRun:
    """
    Record of a complete optimization run.

    Stores configuration, best results, and all trial data.
    """

    run_id: str
    study_name: str
    started_at: datetime
    completed_at: datetime | None
    config: OptimizationConfig
    best_parameters: dict[str, Any] | None
    best_objective_value: float | None
    total_trials: int
    feasible_trials: int
    trials: list[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Custom serialization to handle datetimes and Decimals
        return {
            "run_id": self.run_id,
            "study_name": self.study_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self._serialize_config(self.config),
            "best_parameters": self.best_parameters,
            "best_objective_value": self.best_objective_value,
            "total_trials": self.total_trials,
            "feasible_trials": self.feasible_trials,
            "trials": [self._serialize_trial(t) for t in self.trials],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationRun:
        """Create from dictionary."""
        # Note: Deserialization of complex nested objects like config and trials
        # is simplified here. Full reconstruction might require more logic
        # if we need to restore exact objects. For analysis, dicts are often enough.
        # For now, we'll load basic fields and leave complex ones as dicts or None
        # if strict typing is needed.
        # However, to be useful, we should try to reconstruct what we can.

        # This is a placeholder for full deserialization logic.
        # Implementing full deserialization requires reconstructing ParameterSpace, etc.
        # which is complex. For storage/viewing, JSON is fine.
        # If we need to resume, we might need more.

        return cls(
            run_id=data["run_id"],
            study_name=data["study_name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            config=data["config"],  # Keeping as dict for now
            best_parameters=data.get("best_parameters"),
            best_objective_value=data.get("best_objective_value"),
            total_trials=data["total_trials"],
            feasible_trials=data["feasible_trials"],
            trials=[],  # Skip loading full trials for lightweight object, or load if needed
        )

    def _serialize_config(self, config: OptimizationConfig | dict[str, Any]) -> dict[str, Any]:
        """Serialize configuration."""
        # Handle case where config is already a dict (loaded from storage)
        if isinstance(config, dict):
            return config

        # Simplified serialization for OptimizationConfig objects
        return {
            "study_name": config.study_name,
            "objective_name": config.objective_name,
            "direction": config.direction,
            "number_of_trials": config.number_of_trials,
            "sampler_type": config.sampler_type,
            # Parameter space is too complex to serialize fully here without custom logic
            "parameter_count": config.parameter_space.parameter_count,
        }

    def _serialize_trial(self, trial: TrialResult) -> dict[str, Any]:
        """Serialize trial result."""
        return {
            "trial_number": trial.trial_number,
            "parameters": trial.parameters,
            "objective_value": trial.objective_value,
            "is_feasible": trial.is_feasible,
            "duration_seconds": trial.duration_seconds,
            # Skip full backtest result to save space, or include summary
            "metrics": (
                {
                    "total_return": (
                        str(trial.risk_metrics.total_return_pct) if trial.risk_metrics else None
                    ),
                    "sharpe": str(trial.risk_metrics.sharpe_ratio) if trial.risk_metrics else None,
                    "drawdown": (
                        str(trial.risk_metrics.max_drawdown_pct) if trial.risk_metrics else None
                    ),
                    "trades": (
                        trial.trade_statistics.total_trades if trial.trade_statistics else None
                    ),
                }
                if trial.risk_metrics
                else None
            ),
        }


class OptimizationStorage:
    """
    Manages persistence of optimization runs.

    Saves results to ~/.gpt_trader/optimize/
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize storage.

        Args:
            base_dir: Base directory for storage. Defaults to ~/.gpt_trader/optimize
        """
        if base_dir is None:
            self.base_dir = Path.home() / ".gpt_trader" / "optimize"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: OptimizationRun) -> Path:
        """
        Save optimization run to disk.

        Args:
            run: OptimizationRun object

        Returns:
            Path to saved file
        """
        run_dir = self.base_dir / run.run_id
        run_dir.mkdir(exist_ok=True)

        file_path = run_dir / "results.json"

        with open(file_path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

        logger.info(f"Saved optimization run {run.run_id} to {file_path}")
        return file_path

    def load_run(self, run_id: str) -> OptimizationRun | None:
        """
        Load optimization run from disk.

        Args:
            run_id: Run identifier

        Returns:
            OptimizationRun object or None if not found
        """
        file_path = self.base_dir / run_id / "results.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)
                return OptimizationRun.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load run {run_id}: {e}")
            return None

    def list_runs(self) -> list[dict[str, Any]]:
        """
        List all saved runs.

        Returns:
            List of run summaries
        """
        runs = []
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir():
                file_path = run_dir / "results.json"
                if file_path.exists():
                    try:
                        with open(file_path) as f:
                            # Read only start of file for metadata if possible,
                            # but JSON requires full load. For large files this is slow.
                            # Optimization: Store metadata.json separately.
                            # For now, just load and catch errors.
                            data = json.load(f)
                            runs.append(
                                {
                                    "run_id": data["run_id"],
                                    "study_name": data["study_name"],
                                    "started_at": data["started_at"],
                                    "best_value": data.get("best_objective_value"),
                                }
                            )
                    except Exception:
                        continue

        # Sort by start time descending
        runs.sort(key=lambda x: x["started_at"], reverse=True)
        return runs
