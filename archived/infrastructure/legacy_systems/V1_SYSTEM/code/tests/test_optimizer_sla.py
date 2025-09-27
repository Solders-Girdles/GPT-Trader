from __future__ import annotations

import os
import time

from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.portfolio.optimizer import PortfolioConstraints, PortfolioOptimizer


def _make_strategy(idx: int) -> StrategyMetadata:
    ctx = StrategyContext(
        market_regime="trending",
        time_period="bull_market",
        asset_class="equity",
        risk_profile="moderate",
        volatility_regime="medium",
        correlation_regime="medium",
    )
    perf = StrategyPerformance(
        sharpe_ratio=1.0 + 0.1 * (idx % 5),
        cagr=0.10 + 0.01 * idx,
        max_drawdown=0.10 + 0.005 * (idx % 3),
        win_rate=0.55,
        consistency_score=0.7,
        n_trades=100 + idx,
        avg_trade_duration=5.0,
        profit_factor=1.4,
        calmar_ratio=1.2,
        sortino_ratio=1.4,
        information_ratio=0.5,
        beta=0.9,
        alpha=0.02,
    )
    import datetime as dt

    return StrategyMetadata(
        strategy_id=f"s{idx}",
        name=f"strategy_{idx}",
        description="",
        strategy_type="trend_following",
        parameters={"p": idx},
        context=ctx,
        performance=perf,
        discovery_date=dt.datetime.now(),
        last_updated=dt.datetime.now(),
        usage_count=0,
        success_rate=0.0,
        tags=[],
        notes="",
    )


def test_optimizer_runtime_sla_under_turnover_constraints():
    # Create a modest set of strategies
    strategies = [_make_strategy(i) for i in range(10)]

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.5,
        max_volatility=0.6,
        transaction_cost_bps=25.0,
        max_turnover=0.4,
    )
    opt = PortfolioOptimizer(constraints=constraints)

    # Baseline run
    alloc = opt.optimize_portfolio(strategies)
    prev: dict[str, float] = alloc.strategy_weights

    # SLA: Re-optimization with prev_weights should complete within threshold seconds
    # Threshold is 2.0s locally; relaxed on CI/low-core runners or via OPTIMIZER_SLA_SEC env
    def _sla_threshold_seconds() -> float:
        # explicit override
        env_val = os.environ.get("OPTIMIZER_SLA_SEC") or os.environ.get("OPT_SLA_SEC")
        if env_val:
            try:
                return float(env_val)
            except Exception:
                pass
        thr = 2.0
        # Relax on CI where runners can be slower
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
            thr = max(thr, 3.5)
        # Relax on very low core count
        try:
            cores = os.cpu_count() or 2
            if cores <= 2:
                thr = max(thr, 3.5)
        except Exception:
            pass
        return thr

    t0 = time.time()
    _ = opt.optimize_portfolio(strategies, prev_weights=prev)
    dt = time.time() - t0

    assert dt < _sla_threshold_seconds(), f"Re-optimization exceeded SLA: {dt:.3f}s"
