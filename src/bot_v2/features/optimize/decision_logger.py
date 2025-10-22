"""
Decision logging for production-parity backtesting.

Provides JSON-based logging of all strategy decisions, enabling:
- Decision parity validation between backtest and live
- Decision replay for debugging
- Strategy behavior analysis
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bot_v2.features.live_trade.strategies.decisions import Decision
from bot_v2.utilities.logging_patterns import get_logger

from .types_v2 import BacktestResult, DecisionContext, DecisionRecord, ExecutionResult

logger = get_logger(__name__, component="optimize")


class DecisionLogger:
    """Logs strategy decisions with full context for later analysis."""

    def __init__(self, *, enabled: bool = True, base_directory: str = "backtesting/decision_logs"):
        """
        Initialize decision logger.

        Args:
            enabled: Whether to actually log decisions
            base_directory: Base directory for decision logs
        """
        self.enabled = enabled
        self.base_directory = Path(base_directory)
        self.decisions: list[DecisionRecord] = []

    def log_decision(
        self,
        *,
        context: DecisionContext,
        decision: Decision,
        execution: ExecutionResult,
    ) -> None:
        """
        Log a single decision with its context and execution result.

        Args:
            context: Market context at decision time
            decision: Strategy decision
            execution: Simulated execution result
        """
        if not self.enabled:
            return

        record = DecisionRecord(context=context, decision=decision, execution=execution)
        self.decisions.append(record)

    def save(self, result: BacktestResult) -> Path:
        """
        Save backtest result with all decisions to JSON file.

        Args:
            result: Complete backtest result

        Returns:
            Path to saved file
        """
        if not self.enabled:
            logger.warning("Decision logging disabled, skipping save")
            return Path()

        # Create directory structure: base_dir/YYYY-MM-DD/
        date_dir = self.base_directory / result.start_time.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Filename: bt_{timestamp}_{symbol}.json
        timestamp_str = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"bt_{timestamp_str}_{result.symbol}.json"
        filepath = date_dir / filename

        # Serialize result
        data = result.to_dict()

        # Write with pretty formatting for human readability
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Decision log saved | path=%s | decisions=%d | trades=%d",
            filepath,
            len(result.decisions),
            result.metrics.total_trades if result.metrics else 0,
        )

        return filepath

    def load(self, filepath: Path) -> BacktestResult:
        """
        Load backtest result from JSON file.

        Args:
            filepath: Path to decision log file

        Returns:
            BacktestResult with all decisions

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(filepath) as f:
            data = json.load(f)

        return self._deserialize_result(data)

    def _deserialize_result(self, data: dict[str, Any]) -> BacktestResult:
        """Deserialize BacktestResult from dict."""
        from .types import BacktestMetrics
        from .types_v2 import BacktestConfig

        # Reconstruct config
        config_data = data["config"]
        from decimal import Decimal

        config = BacktestConfig(
            initial_capital=Decimal(config_data["initial_capital"]),
            commission_rate=Decimal(config_data["commission_rate"]),
            slippage_rate=Decimal(config_data["slippage_rate"]),
            enable_decision_logging=config_data["enable_decision_logging"],
            log_directory=config_data["log_directory"],
        )

        # Reconstruct decisions
        decisions = [DecisionRecord.from_dict(d) for d in data["decisions"]]

        # Reconstruct metrics
        metrics = None
        if data["metrics"]:
            m = data["metrics"]
            metrics = BacktestMetrics(
                total_return=m["total_return"],
                sharpe_ratio=m["sharpe_ratio"],
                max_drawdown=m["max_drawdown"],
                win_rate=m["win_rate"],
                profit_factor=m["profit_factor"],
                total_trades=m["total_trades"],
                avg_trade=m["avg_trade"],
                best_trade=m["best_trade"],
                worst_trade=m["worst_trade"],
                recovery_factor=m["recovery_factor"],
                calmar_ratio=m["calmar_ratio"],
            )

        # Reconstruct equity curve
        equity_curve = [
            (datetime.fromisoformat(ts), Decimal(eq)) for ts, eq in data["equity_curve"]
        ]

        return BacktestResult(
            run_id=data["run_id"],
            strategy_name=data["strategy_name"],
            symbol=data["symbol"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            config=config,
            decisions=decisions,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    def get_decision_count(self) -> int:
        """Get count of logged decisions."""
        return len(self.decisions)

    def get_execution_count(self) -> int:
        """Get count of executed trades (filled decisions)."""
        return sum(1 for d in self.decisions if d.execution.filled)

    def clear(self) -> None:
        """Clear logged decisions (for new backtest run)."""
        self.decisions.clear()


def load_decision_log(filepath: Path) -> BacktestResult:
    """
    Convenience function to load a decision log.

    Args:
        filepath: Path to decision log JSON file

    Returns:
        BacktestResult with all decisions
    """
    logger_instance = DecisionLogger(enabled=True)
    return logger_instance.load(filepath)


def compare_decision_logs(
    backtest_log: Path,
    live_log: Path,
) -> dict[str, Any]:
    """
    Compare decisions from backtest vs live execution.

    Args:
        backtest_log: Path to backtest decision log
        live_log: Path to live decision log

    Returns:
        Comparison report with differences
    """
    bt_result = load_decision_log(backtest_log)
    live_result = load_decision_log(live_log)

    # Compare decision counts
    bt_decisions = len(bt_result.decisions)
    live_decisions = len(live_result.decisions)

    # Find action mismatches
    mismatches = []
    min_len = min(bt_decisions, live_decisions)

    for i in range(min_len):
        bt_dec = bt_result.decisions[i]
        live_dec = live_result.decisions[i]

        if bt_dec.decision.action != live_dec.decision.action:
            mismatches.append(
                {
                    "index": i,
                    "timestamp": bt_dec.context.timestamp.isoformat(),
                    "backtest_action": bt_dec.decision.action.value,
                    "live_action": live_dec.decision.action.value,
                    "backtest_reason": bt_dec.decision.reason,
                    "live_reason": live_dec.decision.reason,
                }
            )

    return {
        "backtest_decisions": bt_decisions,
        "live_decisions": live_decisions,
        "decision_count_match": bt_decisions == live_decisions,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "parity_rate": 1.0 - (len(mismatches) / min_len) if min_len > 0 else 0.0,
    }


__all__ = ["DecisionLogger", "load_decision_log", "compare_decision_logs"]
