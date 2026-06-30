"""Walk-forward analysis data models.

Configuration and result contracts for walk-forward optimization: window
definitions, per-window results, and the aggregate result. The optimizer that
produces them lives in walk_forward.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult


@dataclass
class WalkForwardConfig:
    """Configuration for Walk-Forward Analysis.

    Attributes:
        train_months: Duration of training (in-sample) window in months
        test_months: Duration of test (out-of-sample) window in months
        anchor_start: If True, use anchored (expanding) window; if False, use rolling window
        min_trades_per_window: Minimum trades required in each window for valid results
        overlap_months: Overlap between consecutive train windows (default: 0)
    """

    train_months: int = 6
    test_months: int = 1
    anchor_start: bool = False
    min_trades_per_window: int = 10
    overlap_months: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.train_months < 1:
            raise ValueError("train_months must be at least 1")
        if self.test_months < 1:
            raise ValueError("test_months must be at least 1")
        if self.overlap_months >= self.train_months:
            raise ValueError("overlap_months must be less than train_months")


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_days(self) -> int:
        """Number of days in training period."""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """Number of days in test period."""
        return (self.test_end - self.test_start).days


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""

    window: WalkForwardWindow
    best_parameters: dict[str, Any]
    optimization_trials: int
    best_train_objective: float

    # Out-of-sample (test) results
    test_result: BacktestResult | None = None
    test_risk_metrics: RiskMetrics | None = None
    test_trade_stats: TradeStatistics | None = None
    test_objective_value: float | None = None

    # Validation
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated results from Walk-Forward Analysis."""

    # Configuration
    config: WalkForwardConfig
    total_windows: int
    valid_windows: int

    # Aggregated out-of-sample metrics
    aggregate_return_pct: Decimal
    aggregate_sharpe: Decimal | None
    aggregate_max_drawdown_pct: Decimal
    total_trades: int
    overall_win_rate: Decimal

    # Individual window results
    window_results: list[WindowResult]

    # Robustness metrics
    parameter_stability: dict[str, float]  # Variance of each parameter across windows
    performance_consistency: float  # Correlation between train and test performance

    # Time tracking
    total_duration_seconds: float
