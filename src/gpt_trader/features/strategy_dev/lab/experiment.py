"""Experiment definition and tracking for strategy development.

Provides:
- Experiment: Define and track strategy experiments
- ExperimentResult: Capture results with metrics
- ExperimentStatus: Track experiment lifecycle
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class ExperimentStatus(Enum):
    """Status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentResult:
    """Results from a completed experiment.

    Captures all metrics and metadata from running an experiment.
    """

    # Core metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float

    # Time-based metrics
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Additional metrics
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0

    # Regime-specific performance
    regime_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    # Raw data
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    notes: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "expectancy": self.expectancy,
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "average_win": self.average_win,
                "average_loss": self.average_loss,
            },
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": self.duration_seconds,
            },
            "regime_performance": self.regime_performance,
            "equity_curve_length": len(self.equity_curve),
            "trade_count": len(self.trade_log),
            "notes": self.notes,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        metrics = data.get("metrics", {})
        trades = data.get("trades", {})
        timing = data.get("timing", {})

        return cls(
            total_return=metrics.get("total_return", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            calmar_ratio=metrics.get("calmar_ratio", 0.0),
            expectancy=metrics.get("expectancy", 0.0),
            total_trades=trades.get("total", 0),
            winning_trades=trades.get("winning", 0),
            losing_trades=trades.get("losing", 0),
            average_win=trades.get("average_win", 0.0),
            average_loss=trades.get("average_loss", 0.0),
            start_time=datetime.fromisoformat(timing.get("start_time", datetime.now().isoformat())),
            end_time=datetime.fromisoformat(timing.get("end_time", datetime.now().isoformat())),
            duration_seconds=timing.get("duration_seconds", 0.0),
            regime_performance=data.get("regime_performance", {}),
            notes=data.get("notes", ""),
            error_message=data.get("error_message"),
        )

    def compare_to(self, other: "ExperimentResult") -> dict[str, float]:
        """Compare this result to another, returning percentage differences."""
        if other.total_return == 0:
            return_diff = float("inf") if self.total_return != 0 else 0.0
        else:
            return_diff = ((self.total_return - other.total_return) / abs(other.total_return)) * 100

        if other.sharpe_ratio == 0:
            sharpe_diff = float("inf") if self.sharpe_ratio != 0 else 0.0
        else:
            sharpe_diff = ((self.sharpe_ratio - other.sharpe_ratio) / abs(other.sharpe_ratio)) * 100

        if other.max_drawdown == 0:
            drawdown_diff = float("inf") if self.max_drawdown != 0 else 0.0
        else:
            # For drawdown, lower is better, so we invert
            drawdown_diff = (
                (other.max_drawdown - self.max_drawdown) / abs(other.max_drawdown)
            ) * 100

        return {
            "return_improvement_percent": return_diff,
            "sharpe_improvement_percent": sharpe_diff,
            "drawdown_improvement_percent": drawdown_diff,
            "win_rate_difference": self.win_rate - other.win_rate,
            "trade_count_difference": self.total_trades - other.total_trades,
        }


@dataclass
class Experiment:
    """An experiment for testing strategy variations.

    Experiments track:
    - Strategy parameters being tested
    - Baseline for comparison
    - Results and metrics
    - Status and lifecycle
    """

    name: str
    description: str

    # Strategy configuration
    strategy_name: str
    parameters: dict[str, Any]

    # Experiment settings
    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Baseline comparison
    baseline_parameters: dict[str, Any] | None = None
    baseline_result: ExperimentResult | None = None

    # Status tracking
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: ExperimentResult | None = None

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    # Version tracking
    version: int = 1
    parent_experiment_id: str | None = None

    def __post_init__(self) -> None:
        """Generate parameter hash for deduplication."""
        self._parameter_hash = self._compute_parameter_hash()

    def _compute_parameter_hash(self) -> str:
        """Compute hash of parameters for deduplication."""
        param_str = json.dumps(self.parameters, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:12]

    @property
    def parameter_hash(self) -> str:
        """Get parameter hash."""
        return self._parameter_hash

    def start(self) -> None:
        """Mark experiment as started."""
        if self.status != ExperimentStatus.PENDING:
            raise ValueError(f"Cannot start experiment in {self.status} status")
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, result: ExperimentResult) -> None:
        """Mark experiment as completed with result."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot complete experiment in {self.status} status")
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def fail(self, error_message: str) -> None:
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.now()
        if self.result is None:
            # Create minimal failed result
            now = datetime.now()
            self.result = ExperimentResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                average_win=0.0,
                average_loss=0.0,
                start_time=self.started_at or now,
                end_time=now,
                duration_seconds=0.0,
                error_message=error_message,
            )
        else:
            self.result.error_message = error_message

    def cancel(self) -> None:
        """Cancel the experiment."""
        if self.status == ExperimentStatus.COMPLETED:
            raise ValueError("Cannot cancel completed experiment")
        self.status = ExperimentStatus.CANCELLED
        self.completed_at = datetime.now()

    def clone(self, new_parameters: dict[str, Any] | None = None) -> "Experiment":
        """Create a new experiment based on this one."""
        params = new_parameters if new_parameters is not None else self.parameters.copy()

        return Experiment(
            name=f"{self.name}_v{self.version + 1}",
            description=self.description,
            strategy_name=self.strategy_name,
            parameters=params,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
            baseline_parameters=self.baseline_parameters,
            baseline_result=self.baseline_result,
            tags=self.tags.copy(),
            version=self.version + 1,
            parent_experiment_id=self.experiment_id,
        )

    def get_comparison(self) -> dict[str, Any] | None:
        """Get comparison with baseline if available."""
        if self.result is None or self.baseline_result is None:
            return None
        return self.result.compare_to(self.baseline_result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "parameter_hash": self.parameter_hash,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "baseline_parameters": self.baseline_parameters,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
            "baseline_result": self.baseline_result.to_dict() if self.baseline_result else None,
            "tags": self.tags,
            "version": self.version,
            "parent_experiment_id": self.parent_experiment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        experiment = cls(
            experiment_id=data.get("experiment_id", str(uuid.uuid4())[:8]),
            name=data["name"],
            description=data.get("description", ""),
            strategy_name=data["strategy_name"],
            parameters=data["parameters"],
            symbol=data.get("symbol", "BTC-USD"),
            timeframe=data.get("timeframe", "1h"),
            start_date=(
                datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None
            ),
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            baseline_parameters=data.get("baseline_parameters"),
            status=ExperimentStatus(data.get("status", "pending")),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            tags=data.get("tags", []),
            version=data.get("version", 1),
            parent_experiment_id=data.get("parent_experiment_id"),
        )

        if data.get("result"):
            experiment.result = ExperimentResult.from_dict(data["result"])

        if data.get("baseline_result"):
            experiment.baseline_result = ExperimentResult.from_dict(data["baseline_result"])

        return experiment

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Experiment(id={self.experiment_id}, name={self.name}, "
            f"status={self.status.value}, strategy={self.strategy_name})"
        )
