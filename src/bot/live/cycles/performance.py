from __future__ import annotations

from datetime import datetime
from typing import Any

from bot.monitor.alerts import AlertSeverity


async def execute_performance_cycle(orchestrator: Any) -> None:
    """Execute one performance monitoring cycle via orchestrator state."""
    logger = orchestrator.__class__.__dict__.get("logger", None) or __import__("logging").getLogger(
        __name__
    )

    logger.info("Executing performance monitoring cycle")

    # Get current performance metrics
    performance_summary = orchestrator.performance_monitor.get_performance_summary()

    # Check for performance issues
    if performance_summary.get("status") != "no_optimization":
        current_sharpe = performance_summary.get("sharpe_ratio", 0)
        current_drawdown = performance_summary.get("max_drawdown", 0)

        # Check Sharpe ratio
        if current_sharpe < orchestrator.config.min_sharpe_ratio:
            await orchestrator.alert_manager.send_performance_alert(
                "portfolio",
                "sharpe_ratio",
                current_sharpe,
                orchestrator.config.min_sharpe_ratio,
                AlertSeverity.WARNING,
            )

        # Check drawdown
        if current_drawdown > orchestrator.config.max_drawdown_threshold:
            await orchestrator.alert_manager.send_performance_alert(
                "portfolio",
                "max_drawdown",
                current_drawdown,
                orchestrator.config.max_drawdown_threshold,
                AlertSeverity.ERROR,
            )

    # Record operation
    perf_summary = dict(performance_summary)
    if "timestamp" not in perf_summary:
        perf_summary["timestamp"] = datetime.now().isoformat()
    orchestrator._record_operation("performance_monitoring", perf_summary)

    logger.info("Performance monitoring cycle completed")

    # Phase 1: compute selection metrics if we have a recent selection snapshot
    try:
        if (
            orchestrator._last_selection_predicted_ranks
            and orchestrator.strategy_selector.current_selection
        ):
            # Use selection's performance_score as a placeholder for realized performance ranking
            sel = orchestrator.strategy_selector.current_selection
            actual_perf = {s.strategy_id: float(s.performance_score) for s in sel}
            selected_ids = orchestrator._last_selection_selected or []
            snapshot = orchestrator.performance_monitor.record_selection_metrics(
                predicted_ranks=orchestrator._last_selection_predicted_ranks,
                actual_performance=actual_perf,
                selected_strategies=selected_ids,
            )
            logger.info(f"Selection metrics: {snapshot}")
    except Exception as e:
        logger.debug(f"Selection metrics recording skipped: {e}")
