from __future__ import annotations

from datetime import datetime
from typing import Any

from bot.live.audit import record_selection_change, record_trade_blocked
from bot.monitor.alerts import AlertSeverity


async def execute_selection_cycle(orchestrator: Any) -> None:
    """Execute one strategy selection and optimization cycle via orchestrator state.

    Mirrors the in-orchestrator implementation but lives in a separate module to
    simplify maintenance as the system grows.
    """
    logger = orchestrator.__class__.__dict__.get("logger", None) or __import__("logging").getLogger(
        __name__
    )

    logger.info("Executing strategy selection cycle")
    try:
        # Get current market data (placeholder for future use)
        await orchestrator._get_current_market_data()

        # Select strategies
        selected_strategies = orchestrator.strategy_selector.get_current_selection()

        if not selected_strategies:
            logger.warning("No strategies selected")
            # Record a no-op selection cycle for observability/tests
            orchestrator._record_operation(
                "strategy_selection",
                {
                    "n_strategies": 0,
                    "status": "no_selection",
                },
            )
            return
    except Exception as e:
        # Handle selection errors gracefully for direct method calls (tests)
        logger.error(f"Error during strategy selection: {e}")
        try:
            await orchestrator.alert_manager.send_system_alert(
                "strategy_selection",
                "error",
                str(e),
                AlertSeverity.ERROR,
            )
        except Exception:
            pass
        return

    # Get strategy metadata
    strategy_metadata = [score.strategy for score in selected_strategies]

    # Optimize portfolio
    prev_weights = (
        orchestrator.portfolio_optimizer.last_optimization.strategy_weights
        if orchestrator.portfolio_optimizer.last_optimization is not None
        else None
    )
    allocation = orchestrator.portfolio_optimizer.optimize_portfolio(
        strategies=strategy_metadata,
        historical_returns=None,  # Would use actual historical data
        prev_weights=prev_weights,
    )

    # Apply Phase 1 safety rails to strategy weights using simple risk scores
    try:
        risk_scores = {
            s.strategy_id: max(0.0, float(getattr(s.performance, "max_drawdown", 0.1)))
            for s in strategy_metadata
        }
        safe_weights = orchestrator.safety_rails.apply_safety_constraints(
            allocation.strategy_weights,
            risk_scores,
        )
        # Update allocation if materially changed
        if any(
            abs(safe_weights.get(k, 0.0) - allocation.strategy_weights.get(k, 0.0)) > 1e-6
            for k in set(allocation.strategy_weights) | set(safe_weights)
        ):
            allocation.strategy_weights = safe_weights

        # Validate and alert on violations (non-fatal)
        ok, violations = orchestrator.safety_rails.validate_allocations(
            allocation.strategy_weights, risk_scores
        )
        if not ok and violations:
            for v in violations:
                logger.warning(f"Safety rails violation: {v}")
                try:
                    await orchestrator.alert_manager.send_risk_alert(
                        risk_type="allocations",
                        current_value=0.0,
                        limit_value=0.0,
                        severity=AlertSeverity.WARNING,
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Safety rails application skipped: {e}")

    # Compute transition smoothness relative to previous allocation
    try:
        prev_weights = (
            orchestrator.portfolio_optimizer.last_optimization.strategy_weights
            if orchestrator.portfolio_optimizer.last_optimization is not None
            else {sid: 0.0 for sid in allocation.strategy_weights.keys()}
        )
        # Align keys
        keys = set(prev_weights) | set(allocation.strategy_weights)
        prev_aligned = {k: prev_weights.get(k, 0.0) for k in keys}
        curr_aligned = {k: allocation.strategy_weights.get(k, 0.0) for k in keys}
        # Turnover (L1) and smoothness
        turnover_rate = orchestrator.transition_calc.calculate_turnover_rate(
            prev_aligned, curr_aligned
        )
        smoothness = orchestrator.transition_calc.calculate_smoothness_score(
            prev_aligned,
            curr_aligned,
            portfolio_value=orchestrator.config.assumed_portfolio_value,
        )
        slippage_cost_estimate = None
        if orchestrator.config.enable_slippage_estimation:
            try:
                slippage_cost_estimate = orchestrator.transition_calc.calculate_slippage_cost(
                    prev_aligned,
                    curr_aligned,
                    portfolio_value=orchestrator.config.assumed_portfolio_value,
                )
            except Exception:
                slippage_cost_estimate = None
        # Expose turnover to performance monitoring
        try:
            if hasattr(orchestrator.performance_monitor, "record_turnover"):
                orchestrator.performance_monitor.record_turnover(float(turnover_rate))
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Transition smoothness calculation failed: {e}")
        smoothness = 0.0
        slippage_cost_estimate = None
        turnover_rate = 0.0

    # Optional alert if smoothness below threshold (prefer monitoring config; fallback to orchestrator config)
    try:
        threshold = None
        try:
            threshold = getattr(orchestrator, "performance_monitor", None).thresholds.min_transition_smoothness  # type: ignore[attr-defined]
        except Exception:
            threshold = None
        if threshold is None:
            threshold = getattr(orchestrator.config, "transition_smoothness_alert_threshold", None)
        if threshold is not None and smoothness < float(threshold):
            await orchestrator.alert_manager.send_strategy_alert(
                strategy_id="portfolio",
                event="low_transition_smoothness",
                details=(f"smoothness={smoothness:.3f} below threshold={float(threshold):.3f}"),
                severity=AlertSeverity.WARNING,
            )
    except Exception:
        pass

    # Observability: log decision details and summary metrics
    try:
        orchestrator.observability.log_decision(
            decision_type="strategy_selection",
            decision_data={
                "selected": [s.strategy_id for s in strategy_metadata],
                "weights": allocation.strategy_weights,
                "smoothness": smoothness,
                **(
                    {"slippage_cost_estimate": float(slippage_cost_estimate)}
                    if isinstance(slippage_cost_estimate, int | float)
                    else {}
                ),
            },
            metadata={
                "mode": orchestrator.config.mode.value,
                "timestamp": datetime.now().isoformat(),
            },
        )
        orchestrator.observability.log_metrics(
            metrics={
                "expected_sharpe": float(allocation.sharpe_ratio),
                "expected_volatility": float(allocation.expected_volatility),
                "transition_smoothness": float(smoothness),
                **(
                    {"slippage_cost_estimate": float(slippage_cost_estimate)}
                    if isinstance(slippage_cost_estimate, int | float)
                    else {}
                ),
            },
            model_version="portfolio",
        )
        # Persist metrics for comparison across runs
        version_tag = f"portfolio_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        orchestrator.metrics_registry.log_metrics(
            model_version=version_tag,
            metrics={
                "expected_sharpe": float(allocation.sharpe_ratio),
                "expected_volatility": float(allocation.expected_volatility),
                "transition_smoothness": float(smoothness),
                **(
                    {"slippage_cost_estimate": float(slippage_cost_estimate)}
                    if isinstance(slippage_cost_estimate, int | float)
                    else {}
                ),
            },
            metadata={
                "mode": orchestrator.config.mode.value,
                "n_strategies": len(strategy_metadata),
                "timestamp": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        logger.debug(f"Observability logging skipped: {e}")

    # Update risk manager with new allocation
    orchestrator.risk_manager.portfolio_allocation = allocation

    # Execute portfolio changes if in automated mode (value-compare to avoid cycle import)
    if getattr(orchestrator.config.mode, "value", str(orchestrator.config.mode)) == "automated":
        # Portfolio-level drawdown guard
        try:
            perf_summary = orchestrator.performance_monitor.get_performance_summary()
            # portfolio entry is stored under strategies key when present
            latest_mdd = None
            try:
                portfolio_data = perf_summary.get("strategies", {}).get("portfolio", {})
                latest_mdd = portfolio_data.get("current_drawdown")
            except Exception:
                latest_mdd = None
            if (
                latest_mdd is not None
                and orchestrator.safety_rails.should_block_trading_due_to_drawdown(
                    float(latest_mdd)
                )
            ):
                logger.warning(
                    f"Trading blocked by drawdown guard: max_drawdown {float(latest_mdd):.3f} > "
                    f"limit {orchestrator.safety_rails.max_drawdown_limit:.3f}"
                )
                await orchestrator.alert_manager.send_risk_alert(
                    risk_type="portfolio_drawdown",
                    current_value=float(latest_mdd),
                    limit_value=float(orchestrator.safety_rails.max_drawdown_limit),
                    severity=AlertSeverity.ERROR,
                )
                # Record blocked operation
                record_trade_blocked(
                    orchestrator,
                    reason="drawdown_guard",
                    details={
                        "current_drawdown": float(latest_mdd),
                        "limit": float(orchestrator.safety_rails.max_drawdown_limit),
                    },
                )
            else:
                await orchestrator._execute_portfolio_changes(allocation)
        except Exception as e:
            logger.warning(f"Drawdown guard check failed, proceeding with execution: {e}")
            await orchestrator._execute_portfolio_changes(allocation)

    # Record operation
    orchestrator._record_operation(
        "strategy_selection",
        {
            "n_strategies": len(selected_strategies),
            "allocation": allocation.strategy_weights,
            "expected_sharpe": allocation.sharpe_ratio,
            "expected_volatility": allocation.expected_volatility,
            "transition_smoothness": smoothness,
            "turnover": float(turnover_rate),
            **(
                {"slippage_cost_estimate": float(slippage_cost_estimate)}
                if isinstance(slippage_cost_estimate, int | float)
                else {}
            ),
        },
    )

    # Cache selection snapshot for later evaluation
    try:
        new_selected = [s.strategy_id for s in selected_strategies]
        predicted = [
            s.strategy_id
            for s in sorted(selected_strategies, key=lambda x: x.overall_score, reverse=True)
        ]
        # Audit selection change if previous exists and changed
        if (
            orchestrator._last_selection_selected is not None
            and new_selected != orchestrator._last_selection_selected
        ):
            record_selection_change(
                orchestrator,
                old_selection=list(orchestrator._last_selection_selected),
                new_selection=new_selected,
            )
        orchestrator._last_selection_predicted_ranks = predicted
        orchestrator._last_selection_selected = new_selected
    except Exception:
        orchestrator._last_selection_predicted_ranks = None
        orchestrator._last_selection_selected = None

    logger.info(f"Strategy selection cycle completed. Sharpe: {allocation.sharpe_ratio:.3f}")
