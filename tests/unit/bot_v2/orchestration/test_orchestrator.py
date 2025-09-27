"""Unit tests for TradingOrchestrator."""

import os

from bot_v2.orchestration.orchestrator import TradingOrchestrator
from bot_v2.orchestration.types import (
    OrchestratorConfig,
    OrchestrationResult,
    TradingMode,
)
from bot_v2.features.optimize.types import BacktestMetrics, OptimizationResult


def test_execute_trading_cycle_uses_mock_provider(monkeypatch):
    # Ensure deterministic mock provider
    monkeypatch.setenv("TESTING", "true")

    orch = TradingOrchestrator()
    res = orch.execute_trading_cycle("AAPL")

    assert isinstance(res, OrchestrationResult)
    assert res.symbol == "AAPL"
    assert isinstance(res.data, dict)
    # Expect data fetch attempt info present
    assert "data_fetched" in res.data
    # Reasonable metrics are populated
    assert "execution_time" in res.metrics


def test_optimize_mode_invokes_optimizer(monkeypatch):
    monkeypatch.setenv("TESTING", "true")

    config = OrchestratorConfig(
        mode=TradingMode.OPTIMIZE,
        symbols=["AAPL"],
        enable_ml_strategy=False,
    )
    orchestrator = TradingOrchestrator(config)

    class DummyOptimizer:
        def __init__(self):
            self.calls = []

        def optimize_strategy(self, strategy, symbol, start_date, end_date, **_):
            self.calls.append((strategy, symbol, start_date, end_date))
            metrics = BacktestMetrics(
                total_return=0.15,
                sharpe_ratio=1.5,
                max_drawdown=0.05,
                win_rate=0.6,
                profit_factor=1.8,
                total_trades=42,
                avg_trade=0.003,
                best_trade=0.02,
                worst_trade=-0.01,
                recovery_factor=3.0,
                calmar_ratio=1.2,
            )
            return OptimizationResult(
                strategy=strategy,
                symbol=symbol,
                period=(start_date, end_date),
                best_params={"lookback": 20},
                best_metrics=metrics,
                all_results=[{"params": {"lookback": 20}, "metrics": metrics, "score": 1.5}],
                optimization_time=1.23,
            )

    dummy_optimizer = DummyOptimizer()
    orchestrator.optimizer = dummy_optimizer
    orchestrator.available_slices['optimize'] = True

    result = orchestrator.execute_trading_cycle("AAPL")

    assert dummy_optimizer.calls, "optimize_strategy was not invoked"
    strategy_used, symbol_used, _, _ = dummy_optimizer.calls[0]
    assert strategy_used == "Momentum"
    assert symbol_used == "AAPL"

    assert isinstance(result, OrchestrationResult)
    assert "optimization" in result.data
    assert result.metrics.get("optimize_combinations") == 1
    assert result.metrics.get("optimize_time") == 1.23
    assert not any("Optimization slice not available" in err for err in result.errors)
