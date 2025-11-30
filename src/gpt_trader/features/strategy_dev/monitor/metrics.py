"""Metrics collection and aggregation for performance monitoring.

Provides:
- PerformanceSnapshot: Point-in-time performance capture
- MetricsAggregator: Collect and aggregate metrics over time
"""

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot.

    Captures all relevant metrics at a specific moment.
    """

    timestamp: datetime
    equity: Decimal
    cash: Decimal
    positions_value: Decimal

    # Return metrics
    total_return: float
    daily_return: float
    unrealized_pnl: float
    realized_pnl: float

    # Risk metrics
    drawdown: float
    max_drawdown: float
    volatility: float

    # Trade metrics
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Current regime
    current_regime: str = "UNKNOWN"
    regime_confidence: float = 0.0

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio": {
                "equity": str(self.equity),
                "cash": str(self.cash),
                "positions_value": str(self.positions_value),
            },
            "returns": {
                "total_return": self.total_return,
                "daily_return": self.daily_return,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl": self.realized_pnl,
            },
            "risk": {
                "drawdown": self.drawdown,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
            },
            "trades": {
                "open_positions": self.open_positions,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
            },
            "regime": {
                "current": self.current_regime,
                "confidence": self.regime_confidence,
            },
            "metadata": self.metadata,
        }


@dataclass
class TradeRecord:
    """Record of a single trade."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime | None = None
    pnl: float = 0.0
    regime_at_entry: str = "UNKNOWN"
    regime_at_exit: str | None = None

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class MetricsAggregator:
    """Aggregate and analyze performance metrics over time.

    Features:
    - Rolling window calculations
    - Period-based aggregation (hourly, daily, weekly)
    - Regime-specific performance tracking
    - Trade statistics
    """

    window_size: int = 1000
    initial_equity: Decimal = Decimal("10000")

    # Rolling data stores
    _snapshots: deque = field(default_factory=lambda: deque(maxlen=1000))
    _trades: list[TradeRecord] = field(default_factory=list)
    _daily_returns: deque = field(default_factory=lambda: deque(maxlen=252))
    _regime_performance: dict[str, list[float]] = field(default_factory=dict)

    # Peak tracking
    _peak_equity: Decimal = field(default_factory=lambda: Decimal("10000"))
    _current_equity: Decimal = field(default_factory=lambda: Decimal("10000"))

    def __post_init__(self) -> None:
        """Initialize with proper maxlen."""
        self._snapshots = deque(maxlen=self.window_size)
        self._daily_returns = deque(maxlen=252)
        self._peak_equity = self.initial_equity
        self._current_equity = self.initial_equity

    def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Record a performance snapshot.

        Args:
            snapshot: Performance snapshot to record
        """
        self._snapshots.append(snapshot)
        self._current_equity = snapshot.equity

        # Update peak
        if snapshot.equity > self._peak_equity:
            self._peak_equity = snapshot.equity

        # Track regime performance
        if snapshot.current_regime not in self._regime_performance:
            self._regime_performance[snapshot.current_regime] = []

        if snapshot.daily_return != 0:
            self._regime_performance[snapshot.current_regime].append(snapshot.daily_return)
            self._daily_returns.append(snapshot.daily_return)

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade.

        Args:
            trade: Trade record to add
        """
        self._trades.append(trade)

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return 0.0
        return float((self._peak_equity - self._current_equity) / self._peak_equity)

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown over history."""
        if not self._snapshots:
            return 0.0
        return max(s.drawdown for s in self._snapshots)

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
        """Calculate Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate
            annualize: Whether to annualize the ratio

        Returns:
            Sharpe ratio
        """
        if len(self._daily_returns) < 2:
            return 0.0

        returns = list(self._daily_returns)
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return

        if annualize:
            sharpe *= math.sqrt(252)

        return sharpe

    def get_sortino_ratio(self, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
        """Calculate Sortino ratio (downside risk adjusted).

        Args:
            risk_free_rate: Annual risk-free rate
            annualize: Whether to annualize the ratio

        Returns:
            Sortino ratio
        """
        if len(self._daily_returns) < 2:
            return 0.0

        returns = list(self._daily_returns)
        mean_return = statistics.mean(returns)

        # Only consider negative returns for downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = math.sqrt(sum(r**2 for r in negative_returns) / len(returns))

        if downside_std == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        sortino = (mean_return - daily_rf) / downside_std

        if annualize:
            sortino *= math.sqrt(252)

        return sortino

    def get_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        max_dd = self.get_max_drawdown()
        if max_dd == 0:
            return 0.0

        total_return = self.get_total_return()
        return total_return / max_dd

    def get_total_return(self) -> float:
        """Calculate total return from initial equity."""
        if self.initial_equity == 0:
            return 0.0
        return float((self._current_equity - self.initial_equity) / self.initial_equity)

    def get_volatility(self, annualize: bool = True) -> float:
        """Calculate return volatility.

        Args:
            annualize: Whether to annualize volatility

        Returns:
            Volatility (standard deviation of returns)
        """
        if len(self._daily_returns) < 2:
            return 0.0

        vol = statistics.stdev(self._daily_returns)
        if annualize:
            vol *= math.sqrt(252)
        return vol

    def get_trade_statistics(self) -> dict[str, Any]:
        """Get comprehensive trade statistics."""
        closed_trades = [t for t in self._trades if t.is_closed]

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
            }

        winners = [t for t in closed_trades if t.is_winner]
        losers = [t for t in closed_trades if not t.is_winner]

        total_wins = sum(t.pnl for t in winners) if winners else 0
        total_losses = abs(sum(t.pnl for t in losers)) if losers else 0

        avg_win = total_wins / len(winners) if winners else 0
        avg_loss = total_losses / len(losers) if losers else 0

        win_rate = len(winners) / len(closed_trades)
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "largest_win": max(t.pnl for t in winners) if winners else 0,
            "largest_loss": min(t.pnl for t in losers) if losers else 0,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "total_pnl": sum(t.pnl for t in closed_trades),
        }

    def get_regime_performance(self) -> dict[str, dict[str, float]]:
        """Get performance breakdown by regime."""
        result = {}

        for regime, returns in self._regime_performance.items():
            if not returns:
                continue

            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0

            result[regime] = {
                "sample_count": len(returns),
                "mean_return": mean_return,
                "std_return": std_return,
                "total_return": sum(returns),
                "sharpe": (mean_return / std_return * math.sqrt(252)) if std_return > 0 else 0,
                "win_rate": len([r for r in returns if r > 0]) / len(returns),
            }

        return result

    def get_period_returns(self, period: str = "daily") -> list[tuple[datetime, float]]:
        """Get returns aggregated by period.

        Args:
            period: Aggregation period (daily, weekly, monthly)

        Returns:
            List of (date, return) tuples
        """
        if not self._snapshots:
            return []

        # Group snapshots by period
        period_returns: dict[str, list[float]] = {}

        for snapshot in self._snapshots:
            if period == "daily":
                key = snapshot.timestamp.strftime("%Y-%m-%d")
            elif period == "weekly":
                # Get start of week
                start = snapshot.timestamp - timedelta(days=snapshot.timestamp.weekday())
                key = start.strftime("%Y-%m-%d")
            elif period == "monthly":
                key = snapshot.timestamp.strftime("%Y-%m")
            else:
                key = snapshot.timestamp.strftime("%Y-%m-%d")

            if key not in period_returns:
                period_returns[key] = []
            if snapshot.daily_return != 0:
                period_returns[key].append(snapshot.daily_return)

        # Calculate cumulative returns per period
        result = []
        for date_str, returns in sorted(period_returns.items()):
            if returns:
                # Compound returns within period
                cumulative = 1.0
                for r in returns:
                    cumulative *= 1 + r
                result.append((datetime.fromisoformat(date_str), cumulative - 1))

        return result

    def get_rolling_metrics(self, window: int = 20) -> dict[str, list[float]]:
        """Calculate rolling window metrics.

        Args:
            window: Rolling window size

        Returns:
            Dict of metric names to rolling values
        """
        snapshots = list(self._snapshots)
        if len(snapshots) < window:
            return {"sharpe": [], "volatility": [], "drawdown": []}

        rolling_sharpe = []
        rolling_vol = []
        rolling_dd = []

        for i in range(window, len(snapshots)):
            window_snapshots = snapshots[i - window : i]
            returns = [s.daily_return for s in window_snapshots if s.daily_return != 0]

            if len(returns) > 1:
                mean_ret = statistics.mean(returns)
                std_ret = statistics.stdev(returns)
                rolling_sharpe.append(mean_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0)
                rolling_vol.append(std_ret * math.sqrt(252))
            else:
                rolling_sharpe.append(0)
                rolling_vol.append(0)

            rolling_dd.append(max(s.drawdown for s in window_snapshots))

        return {
            "sharpe": rolling_sharpe,
            "volatility": rolling_vol,
            "drawdown": rolling_dd,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "portfolio": {
                "current_equity": str(self._current_equity),
                "initial_equity": str(self.initial_equity),
                "peak_equity": str(self._peak_equity),
            },
            "returns": {
                "total_return": self.get_total_return(),
                "volatility": self.get_volatility(),
            },
            "risk_adjusted": {
                "sharpe_ratio": self.get_sharpe_ratio(),
                "sortino_ratio": self.get_sortino_ratio(),
                "calmar_ratio": self.get_calmar_ratio(),
            },
            "drawdown": {
                "current": self.get_current_drawdown(),
                "maximum": self.get_max_drawdown(),
            },
            "trades": self.get_trade_statistics(),
            "regime_performance": self.get_regime_performance(),
            "snapshots_recorded": len(self._snapshots),
        }

    def reset(self, initial_equity: Decimal | None = None) -> None:
        """Reset all metrics.

        Args:
            initial_equity: New initial equity (uses current if not provided)
        """
        if initial_equity:
            self.initial_equity = initial_equity

        self._snapshots.clear()
        self._trades.clear()
        self._daily_returns.clear()
        self._regime_performance.clear()
        self._peak_equity = self.initial_equity
        self._current_equity = self.initial_equity
