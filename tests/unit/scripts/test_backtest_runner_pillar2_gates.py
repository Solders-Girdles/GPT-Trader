from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest
from scripts import backtest_runner

from gpt_trader.app.config.bot_config import MeanReversionConfig
from gpt_trader.features.live_trade.strategies.mean_reversion.strategy import MeanReversionStrategy


def _quality_report(*, acceptable: bool = True, has_error: bool = False) -> SimpleNamespace:
    issues = [SimpleNamespace(severity="error")] if has_error else []
    return SimpleNamespace(is_acceptable=acceptable, all_issues=issues)


@pytest.mark.parametrize(
    ("candles_loaded", "total_trades", "expected_threshold", "expected_pass"),
    [
        (72, 0, 1, False),
        (72, 1, 1, True),
        (73, 1, 2, False),
        (73, 2, 2, True),
        (1078, 14, 15, False),
        (1078, 15, 15, True),
    ],
)
def test_pillar_2_trade_gate_scales_with_candles(
    candles_loaded: int,
    total_trades: int,
    expected_threshold: int,
    expected_pass: bool,
) -> None:
    risk_metrics = SimpleNamespace(
        max_drawdown_pct=Decimal("0.5"),
        sharpe_ratio=Decimal("2.0"),
    )
    trade_statistics = SimpleNamespace(
        profit_factor=Decimal("2.0"),
        net_profit_factor=Decimal("1.2"),
        fee_drag_per_trade=Decimal("10"),
        total_trades=total_trades,
    )
    broker_stats = {"total_fees_paid": Decimal("100")}

    gates = backtest_runner._evaluate_pillar_2_gates(
        risk_metrics=risk_metrics,
        trade_statistics=trade_statistics,
        broker_stats=broker_stats,
        initial_equity=Decimal("100000"),
        quality_report=_quality_report(),
        candles_loaded=candles_loaded,
    )

    assert gates["total_trades"]["threshold"] == expected_threshold
    assert gates["total_trades"]["pass"] is expected_pass
    assert gates["trades_per_100_bars"]["pass"] is expected_pass


def test_cli_defaults_include_pillar_2_assumptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["backtest_runner"])
    args = backtest_runner._parse_args()

    assert args.risk_free_rate == 0.0
    assert args.spike_threshold_pct == 15.0
    assert args.volume_anomaly_std == 6.0


def test_lookback_bars_respects_mean_reversion_trend_window() -> None:
    config = MeanReversionConfig(
        lookback_window=20,
        trend_filter_enabled=True,
        trend_window=100,
        trend_threshold_pct=0.01,
    )
    strategy = MeanReversionStrategy(config)

    assert backtest_runner._lookback_bars(strategy) >= 100
