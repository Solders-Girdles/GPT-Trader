"""Main performance monitoring system.

Provides:
- PerformanceMonitor: Unified interface for monitoring strategy performance
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gpt_trader.features.strategy_dev.monitor.alerts import (
    AlertManager,
    AlertRule,
    create_default_alerts,
)
from gpt_trader.features.strategy_dev.monitor.metrics import (
    MetricsAggregator,
    PerformanceSnapshot,
    TradeRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMonitor:
    """Unified performance monitoring system.

    Combines:
    - Real-time metrics collection
    - Alert management
    - Regime-specific analytics
    - Historical data persistence

    Usage:
        monitor = PerformanceMonitor(initial_equity=Decimal("10000"))

        # Add default alerts
        monitor.add_default_alerts()

        # Update on each tick/candle
        monitor.update(
            equity=Decimal("10500"),
            cash=Decimal("5000"),
            positions_value=Decimal("5500"),
            daily_return=0.005,
            current_regime="BULL_QUIET",
        )

        # Get summary
        summary = monitor.get_summary()
    """

    initial_equity: Decimal = Decimal("10000")
    storage_path: Path | None = None
    auto_save_interval: int = 100

    # Components
    metrics: MetricsAggregator = field(default=None)
    alerts: AlertManager = field(default_factory=AlertManager)

    # State tracking
    _snapshot_count: int = 0
    _last_snapshot: PerformanceSnapshot | None = None
    _callbacks: dict[str, list[Callable]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize components."""
        if self.metrics is None:
            self.metrics = MetricsAggregator(initial_equity=self.initial_equity)

        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_state()

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "monitor_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self._snapshot_count = data.get("snapshot_count", 0)
                logger.info(f"Loaded monitor state: {self._snapshot_count} snapshots recorded")
            except Exception as e:
                logger.error(f"Error loading monitor state: {e}")

    def _save_state(self) -> None:
        """Save state to storage."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "monitor_state.json"
        data = {
            "last_updated": datetime.now().isoformat(),
            "snapshot_count": self._snapshot_count,
            "summary": self.metrics.get_summary(),
        }

        with open(state_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_default_alerts(self) -> None:
        """Add default alert rules."""
        for rule in create_default_alerts():
            self.alerts.add_rule(rule)

    def add_alert(self, rule: AlertRule) -> None:
        """Add a custom alert rule.

        Args:
            rule: AlertRule to add
        """
        self.alerts.add_rule(rule)

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for events.

        Events:
        - snapshot: Called on each update with snapshot
        - alert: Called when alert triggers
        - milestone: Called on performance milestones

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

        # Also register alert callbacks
        if event == "alert":
            self.alerts.on_alert(callback)

    def _emit(self, event: str, data: Any) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")

    def update(
        self,
        equity: Decimal,
        cash: Decimal,
        positions_value: Decimal,
        daily_return: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        open_positions: int = 0,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        current_regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PerformanceSnapshot:
        """Update monitor with current state.

        Args:
            equity: Total portfolio equity
            cash: Available cash
            positions_value: Value of open positions
            daily_return: Return for current period
            unrealized_pnl: Unrealized profit/loss
            realized_pnl: Realized profit/loss
            open_positions: Number of open positions
            total_trades: Total trades executed
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            current_regime: Current market regime
            regime_confidence: Confidence in regime detection
            metadata: Additional context

        Returns:
            Created performance snapshot
        """
        # Calculate derived metrics
        total_return = float((equity - self.initial_equity) / self.initial_equity)
        drawdown = (
            float((self.metrics._peak_equity - equity) / self.metrics._peak_equity)
            if self.metrics._peak_equity > 0
            else 0.0
        )
        max_drawdown = max(drawdown, self.metrics.get_max_drawdown())
        volatility = self.metrics.get_volatility(annualize=False)

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=equity,
            cash=cash,
            positions_value=positions_value,
            total_return=total_return,
            daily_return=daily_return,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            drawdown=drawdown,
            max_drawdown=max_drawdown,
            volatility=volatility,
            open_positions=open_positions,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            metadata=metadata or {},
        )

        # Record in metrics
        self.metrics.record_snapshot(snapshot)
        self._snapshot_count += 1
        self._last_snapshot = snapshot

        # Process alerts
        triggered_alerts = self.alerts.process(snapshot)
        for alert in triggered_alerts:
            logger.info(f"Alert triggered: [{alert.severity.value}] {alert.message}")

        # Emit snapshot event
        self._emit("snapshot", snapshot)

        # Check for milestones
        self._check_milestones(snapshot)

        # Auto-save
        if self._snapshot_count % self.auto_save_interval == 0:
            self._save_state()

        return snapshot

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal | None = None,
        entry_time: datetime | None = None,
        exit_time: datetime | None = None,
        pnl: float = 0.0,
        regime_at_entry: str = "UNKNOWN",
        regime_at_exit: str | None = None,
    ) -> None:
        """Record a trade.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price (if closed)
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            pnl: Profit/loss
            regime_at_entry: Regime when trade opened
            regime_at_exit: Regime when trade closed
        """
        trade = TradeRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time or datetime.now(),
            exit_time=exit_time,
            pnl=pnl,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
        )
        self.metrics.record_trade(trade)

    def _check_milestones(self, snapshot: PerformanceSnapshot) -> None:
        """Check for and emit milestone events."""
        # New equity high
        if snapshot.equity > self.metrics._peak_equity:
            self._emit(
                "milestone",
                {
                    "type": "new_high",
                    "equity": str(snapshot.equity),
                    "return": snapshot.total_return,
                },
            )

        # Return milestones
        return_milestones = [0.10, 0.25, 0.50, 1.0, 2.0]
        for milestone in return_milestones:
            # Check if we just crossed this milestone
            if self._last_snapshot:
                prev_return = self._last_snapshot.total_return
                if prev_return < milestone <= snapshot.total_return:
                    self._emit(
                        "milestone",
                        {
                            "type": "return_milestone",
                            "milestone": milestone,
                            "actual_return": snapshot.total_return,
                        },
                    )

    def get_current_snapshot(self) -> PerformanceSnapshot | None:
        """Get most recent snapshot."""
        return self._last_snapshot

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.get_summary()

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary."""
        return self.alerts.get_alert_summary()

    def get_regime_performance(self) -> dict[str, dict[str, float]]:
        """Get performance breakdown by regime."""
        return self.metrics.get_regime_performance()

    def get_trade_statistics(self) -> dict[str, Any]:
        """Get trade statistics."""
        return self.metrics.get_trade_statistics()

    def get_rolling_metrics(self, window: int = 20) -> dict[str, list[float]]:
        """Get rolling window metrics."""
        return self.metrics.get_rolling_metrics(window)

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all monitoring data."""
        return {
            "overview": {
                "initial_equity": str(self.initial_equity),
                "current_equity": str(self.metrics._current_equity),
                "total_return": self.metrics.get_total_return(),
                "snapshots_recorded": self._snapshot_count,
            },
            "metrics": self.metrics.get_summary(),
            "alerts": self.alerts.get_alert_summary(),
            "recent_alerts": [a.to_dict() for a in self.alerts.get_recent_alerts(limit=5)],
        }

    def export_data(self, output_path: Path, format: str = "json") -> None:
        """Export monitoring data.

        Args:
            output_path: Path to export file
            format: Export format (json, csv)
        """
        output_path = Path(output_path)

        data = {
            "exported_at": datetime.now().isoformat(),
            "initial_equity": str(self.initial_equity),
            "summary": self.get_summary(),
            "regime_performance": self.get_regime_performance(),
            "trade_statistics": self.get_trade_statistics(),
        }

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported monitoring data to {output_path}")

    def reset(self, initial_equity: Decimal | None = None) -> None:
        """Reset monitor state.

        Args:
            initial_equity: New initial equity
        """
        if initial_equity:
            self.initial_equity = initial_equity

        self.metrics.reset(initial_equity)
        self.alerts.clear_history()
        self.alerts.reset_all_rules()
        self._snapshot_count = 0
        self._last_snapshot = None

        logger.info("Performance monitor reset")
