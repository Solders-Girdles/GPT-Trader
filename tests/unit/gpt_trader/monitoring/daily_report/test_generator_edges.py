"""Edge-case tests for DailyReportGenerator."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.daily_report.generator as generator_module
from gpt_trader.monitoring.daily_report.generator import DailyReportGenerator
from gpt_trader.monitoring.daily_report.models import DailyReport


@pytest.fixture
def pnl_metrics() -> dict[str, float]:
    return {
        "equity": 100.0,
        "equity_change": 1.0,
        "equity_change_pct": 1.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "funding_pnl": 0.0,
        "total_pnl": 0.0,
        "fees_paid": 0.0,
    }


@pytest.fixture
def trade_metrics() -> dict[str, float | int]:
    return {
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
    }


@pytest.fixture
def risk_metrics() -> dict[str, dict]:
    return {"guard_triggers": {}, "circuit_breaker_state": {}}


@pytest.fixture
def health_metrics() -> dict[str, int]:
    return {
        "stale_marks_count": 0,
        "ws_reconnects": 0,
        "unfilled_orders": 0,
        "api_errors": 0,
    }


@pytest.fixture
def generator_mocks(
    monkeypatch: pytest.MonkeyPatch,
    pnl_metrics: dict[str, float],
    trade_metrics: dict[str, float | int],
    risk_metrics: dict[str, dict],
    health_metrics: dict[str, int],
) -> dict[str, MagicMock]:
    mocks: dict[str, MagicMock] = {
        "load_metrics": MagicMock(return_value={"account": {"equity": 100}}),
        "load_events_since": MagicMock(return_value=[]),
        "calculate_pnl_metrics": MagicMock(return_value=pnl_metrics),
        "calculate_trade_metrics": MagicMock(return_value=trade_metrics),
        "calculate_symbol_metrics": MagicMock(return_value=[]),
        "calculate_risk_metrics": MagicMock(return_value=risk_metrics),
        "calculate_health_metrics": MagicMock(return_value=health_metrics),
        "load_unfilled_orders_count": MagicMock(return_value=0),
        "load_liveness_snapshot": MagicMock(return_value=None),
        "load_runtime_fingerprint": MagicMock(return_value=None),
    }

    for name, mock in mocks.items():
        monkeypatch.setattr(generator_module, name, mock)

    return mocks


@pytest.fixture
def fixed_generated_at(monkeypatch: pytest.MonkeyPatch) -> datetime:
    generated_at = datetime(2024, 2, 2, 14, 0, 0, tzinfo=UTC)

    class FixedDateTime:
        @staticmethod
        def now(_tz=None) -> datetime:
            return generated_at

    monkeypatch.setattr(generator_module, "datetime", FixedDateTime)
    return generated_at


def test_generate_uses_expected_files_and_cutoff(tmp_path, generator_mocks) -> None:
    date = datetime(2024, 2, 2, 12, 30, 0, tzinfo=UTC)
    generator = DailyReportGenerator(profile="alpha", data_dir=tmp_path)
    expected_cutoff = date - timedelta(hours=6)

    report = generator.generate(date=date, lookback_hours=6)

    generator_mocks["load_metrics"].assert_called_once_with(generator.metrics_file)
    generator_mocks["load_events_since"].assert_called_once_with(
        generator.events_file, expected_cutoff
    )
    assert generator_mocks["load_liveness_snapshot"].call_count == 0
    assert generator_mocks["load_runtime_fingerprint"].call_count == 0
    assert report.liveness is None
    assert report.runtime is None
    assert report.date == "2024-02-02"
    assert report.profile == "alpha"


def test_generate_liveness_uses_generated_at(tmp_path, generator_mocks, fixed_generated_at) -> None:
    date = datetime(2024, 2, 2, 12, 30, 0, tzinfo=UTC)
    generator = DailyReportGenerator(profile="alpha", data_dir=tmp_path)
    events_db = tmp_path / "events.db"
    events_db.write_text("db")
    generator_mocks["load_metrics"].return_value = {}

    report = generator.generate(date=date, lookback_hours=6)

    generator_mocks["load_liveness_snapshot"].assert_called_once_with(
        events_db, now=fixed_generated_at
    )
    assert report.generated_at == fixed_generated_at.isoformat()


def test_save_report_writes_json_and_text(tmp_path) -> None:
    generator = DailyReportGenerator(profile="alpha", data_dir=tmp_path)
    report = DailyReport(
        date="2024-02-02",
        profile="alpha",
        generated_at="2024-02-02T00:00:00",
        equity=100.0,
        equity_change=0.0,
        equity_change_pct=0.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        funding_pnl=0.0,
        total_pnl=0.0,
        fees_paid=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        avg_win=0.0,
        avg_loss=0.0,
        largest_win=0.0,
        largest_loss=0.0,
        guard_triggers={},
        circuit_breaker_state={},
        symbol_performance=[],
        stale_marks_count=0,
        ws_reconnects=0,
        unfilled_orders=0,
        api_errors=0,
        liveness=None,
        runtime=None,
    )

    output_dir = tmp_path / "reports"
    text_path = generator.save_report(report, output_dir=output_dir)

    json_path = output_dir / f"daily_report_{report.date}.json"
    assert json_path.exists()
    assert text_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["date"] == report.date
    assert f"Daily Trading Report - {report.date}" in text_path.read_text()
