"""
Types for production-parity backtesting.

These types support backtesting that reuses the production strategy.decide() loop,
ensuring perfect alignment between backtest and live execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.strategies.decisions import Action, Decision

from .types import BacktestMetrics


@dataclass
class DecisionContext:
    """Complete context for a strategy decision at a point in time."""

    timestamp: datetime
    symbol: str
    current_mark: Decimal
    recent_marks: list[Decimal]
    position_state: dict[str, Any] | None
    equity: Decimal
    signal_label: str | None = None
    signal_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_mark": str(self.current_mark),
            "recent_marks": [str(m) for m in self.recent_marks],
            "position_state": self.position_state,
            "equity": str(self.equity),
            "signal_label": self.signal_label,
            "signal_metadata": self.signal_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionContext":
        """Deserialize from dict."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            current_mark=Decimal(data["current_mark"]),
            recent_marks=[Decimal(m) for m in data["recent_marks"]],
            position_state=data["position_state"],
            equity=Decimal(data["equity"]),
            signal_label=data.get("signal_label"),
            signal_metadata=data.get("signal_metadata"),
        )


@dataclass
class ExecutionResult:
    """Result of simulating a decision's execution."""

    filled: bool
    fill_price: Decimal | None = None
    filled_quantity: Decimal | None = None
    commission: Decimal | None = None
    slippage: Decimal | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "filled": self.filled,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "filled_quantity": str(self.filled_quantity) if self.filled_quantity else None,
            "commission": str(self.commission) if self.commission else None,
            "slippage": str(self.slippage) if self.slippage else None,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        """Deserialize from dict."""
        return cls(
            filled=data["filled"],
            fill_price=Decimal(data["fill_price"]) if data.get("fill_price") else None,
            filled_quantity=(
                Decimal(data["filled_quantity"]) if data.get("filled_quantity") else None
            ),
            commission=Decimal(data["commission"]) if data.get("commission") else None,
            slippage=Decimal(data["slippage"]) if data.get("slippage") else None,
            rejection_reason=data.get("rejection_reason"),
        )


@dataclass
class DecisionRecord:
    """Complete record of a single decision and its execution."""

    context: DecisionContext
    decision: Decision
    execution: ExecutionResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "context": self.context.to_dict(),
            "decision": {
                "action": self.decision.action.value,
                "quantity": str(self.decision.quantity) if self.decision.quantity else None,
                "target_notional": (
                    str(self.decision.target_notional) if self.decision.target_notional else None
                ),
                "leverage": self.decision.leverage,
                "reduce_only": self.decision.reduce_only,
                "reason": self.decision.reason,
                "filter_rejected": self.decision.filter_rejected,
                "guard_rejected": self.decision.guard_rejected,
                "rejection_type": self.decision.rejection_type,
            },
            "execution": self.execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        """Deserialize from dict."""
        context = DecisionContext.from_dict(data["context"])
        dec_data = data["decision"]
        decision = Decision(
            action=Action(dec_data["action"]),
            quantity=Decimal(dec_data["quantity"]) if dec_data.get("quantity") else None,
            target_notional=(
                Decimal(dec_data["target_notional"]) if dec_data.get("target_notional") else None
            ),
            leverage=dec_data.get("leverage"),
            reduce_only=dec_data.get("reduce_only", False),
            reason=dec_data.get("reason", ""),
            filter_rejected=dec_data.get("filter_rejected", False),
            guard_rejected=dec_data.get("guard_rejected", False),
            rejection_type=dec_data.get("rejection_type"),
        )
        execution = ExecutionResult.from_dict(data["execution"])
        return cls(context=context, decision=decision, execution=execution)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: Decimal = Decimal("10000")
    commission_rate: Decimal = Decimal("0.001")  # 0.1% (10 bps)
    slippage_rate: Decimal = Decimal("0.0005")  # 0.05% (5 bps)
    enable_decision_logging: bool = True
    log_directory: str = "backtesting/decision_logs"


@dataclass
class BacktestResult:
    """Complete result from a production-parity backtest."""

    run_id: str
    strategy_name: str
    symbol: str
    start_time: datetime
    end_time: datetime
    config: BacktestConfig
    decisions: list[DecisionRecord] = field(default_factory=list)
    metrics: BacktestMetrics | None = None
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "config": {
                "initial_capital": str(self.config.initial_capital),
                "commission_rate": str(self.config.commission_rate),
                "slippage_rate": str(self.config.slippage_rate),
                "enable_decision_logging": self.config.enable_decision_logging,
                "log_directory": self.config.log_directory,
            },
            "decisions": [d.to_dict() for d in self.decisions],
            "metrics": (
                {
                    "total_return": self.metrics.total_return,
                    "sharpe_ratio": self.metrics.sharpe_ratio,
                    "max_drawdown": self.metrics.max_drawdown,
                    "win_rate": self.metrics.win_rate,
                    "profit_factor": self.metrics.profit_factor,
                    "total_trades": self.metrics.total_trades,
                    "avg_trade": self.metrics.avg_trade,
                    "best_trade": self.metrics.best_trade,
                    "worst_trade": self.metrics.worst_trade,
                    "recovery_factor": self.metrics.recovery_factor,
                    "calmar_ratio": self.metrics.calmar_ratio,
                }
                if self.metrics
                else None
            ),
            "equity_curve": [(ts.isoformat(), str(eq)) for ts, eq in self.equity_curve],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.metrics:
            return f"Backtest {self.run_id} - No metrics calculated"

        return f"""
Production-Parity Backtest: {self.run_id}
{'=' * 60}
Strategy: {self.strategy_name}
Symbol: {self.symbol}
Period: {self.start_time.date()} to {self.end_time.date()}
Initial Capital: ${self.config.initial_capital}

Performance Metrics:
- Total Return: {self.metrics.total_return:.2%}
- Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
- Max Drawdown: {self.metrics.max_drawdown:.2%}
- Win Rate: {self.metrics.win_rate:.2%}
- Profit Factor: {self.metrics.profit_factor:.2f}

Trading Activity:
- Total Decisions: {len(self.decisions)}
- Executed Trades: {self.metrics.total_trades}
- Average Trade: {self.metrics.avg_trade:.2%}
- Best Trade: {self.metrics.best_trade:.2%}
- Worst Trade: {self.metrics.worst_trade:.2%}

Risk Metrics:
- Recovery Factor: {self.metrics.recovery_factor:.2f}
- Calmar Ratio: {self.metrics.calmar_ratio:.2f}
        """.strip()


__all__ = [
    "DecisionContext",
    "ExecutionResult",
    "DecisionRecord",
    "BacktestConfig",
    "BacktestResult",
]
