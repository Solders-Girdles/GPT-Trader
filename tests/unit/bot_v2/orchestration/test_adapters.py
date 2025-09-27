"""Unit tests for orchestration adapters and factory."""

from types import SimpleNamespace
import pandas as pd

from bot_v2.orchestration.adapters import (
    DataAdapter,
    AnalyzeAdapter,
    BacktestAdapter,
    OptimizeAdapter,
    MLStrategyAdapter,
    MarketRegimeAdapter,
    PositionSizingAdapter,
    MonitorAdapter,
    TradingAdapter,
    AdapterFactory,
)


def test_data_adapter_gets_data_via_provider():
    # Provider with get_historical_data
    provider = SimpleNamespace(
        get_historical_data=lambda symbol, period: pd.DataFrame({"Close": [1, 2, 3]})
    )
    module = SimpleNamespace(get_data_provider=lambda: provider)
    adapter = DataAdapter(module, "data")
    df = adapter.call("AAPL", period="5d")
    assert df is not None and list(df.columns) == ["Close"] and len(df) == 3


def test_analyze_adapter_prefers_analyze():
    module = SimpleNamespace(
        analyze=lambda data, symbol: {"recommendation": "buy", "confidence": 0.8}
    )
    adapter = AnalyzeAdapter(module, "analyze")
    out = adapter.call("AAPL", pd.DataFrame())
    assert out["recommendation"] == "buy" and out["confidence"] == 0.8


def test_backtest_and_optimize_adapters():
    backtest_mod = SimpleNamespace(run_backtest=lambda symbol, cfg: {"sharpe": 1.2})
    optimize_mod = SimpleNamespace(optimize=lambda symbol, strat, rng: {"best": {"p": 10}})
    bt = BacktestAdapter(backtest_mod, "backtest")
    opt = OptimizeAdapter(optimize_mod, "optimize")
    assert bt.call("AAPL", {"w": 5})["sharpe"] == 1.2
    assert opt.call("AAPL", "ma", {"w": [5, 10]})["best"]["p"] == 10


def test_ml_regime_and_sizing_adapters():
    ml_mod = SimpleNamespace(predict_best_strategy=lambda symbol: {"strategy": "momentum", "confidence": 0.7})
    regime_mod = SimpleNamespace(detect_regime=lambda data, symbol: {"regime": "bull", "confidence": 0.6})
    ml = MLStrategyAdapter(ml_mod, "ml_strategy")
    mr = MarketRegimeAdapter(regime_mod, "market_regime")
    out_ml = ml.call("AAPL")
    out_mr = mr.call("AAPL", pd.DataFrame())
    assert out_ml["strategy"] == "momentum" and out_ml["confidence"] == 0.7
    assert out_mr["regime"] == "bull" and out_mr["confidence"] == 0.6


def test_monitor_and_trading_adapters():
    mon_mod = SimpleNamespace(get_status=lambda: {"status": "ok"}, log_event=lambda e: None)
    trade_mod = SimpleNamespace(execute_live_trade=lambda sym, act, qty, info: {"success": True})
    mon = MonitorAdapter(mon_mod, "monitor")
    tr = TradingAdapter(trade_mod, "live_trade")
    assert mon.call()["status"] == "ok"
    assert tr.call("BTC-PERP", "buy", 1.0, {"s": 1})["success"] is True


def test_adapter_factory_builds_from_registry():
    slices = {
        "data": SimpleNamespace(get_data_provider=lambda: SimpleNamespace(
            get_historical_data=lambda symbol, period: pd.DataFrame({"Close": [1]})
        )),
        "analyze": SimpleNamespace(analyze=lambda data, symbol: {"ok": True}),
        "live_trade": SimpleNamespace(execute_live_trade=lambda s, a, q, i: {"success": True}),
    }

    class Registry:
        def list_available_slices(self):
            return list(slices.keys())

        def get_slice(self, name):
            return slices.get(name)

    adapters = AdapterFactory.create_adapters_for_registry(Registry())
    assert set(adapters.keys()) == {"data", "analyze", "live_trade"}
    assert isinstance(adapters["data"], DataAdapter)
    assert isinstance(adapters["analyze"], AnalyzeAdapter)
    assert isinstance(adapters["live_trade"], TradingAdapter)

