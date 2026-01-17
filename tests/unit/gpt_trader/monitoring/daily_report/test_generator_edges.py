"""Edge-case tests for DailyReportGenerator."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from gpt_trader.monitoring.daily_report.generator import DailyReportGenerator
from gpt_trader.monitoring.daily_report.models import DailyReport


def test_generate_uses_expected_files_and_cutoff(tmp_path) -> None:
    date = datetime(2024, 2, 2, 12, 30, 0, tzinfo=UTC)
    generator = DailyReportGenerator(profile="alpha", data_dir=tmp_path)
    expected_cutoff = date - timedelta(hours=6)

    pnl_metrics = {
        "equity": 100.0,
        "equity_change": 1.0,
        "equity_change_pct": 1.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "funding_pnl": 0.0,
        "total_pnl": 0.0,
        "fees_paid": 0.0,
    }
    trade_metrics = {
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
    risk_metrics = {"guard_triggers": {}, "circuit_breaker_state": {}}
    health_metrics = {
        "stale_marks_count": 0,
        "ws_reconnects": 0,
        "unfilled_orders": 0,
        "api_errors": 0,
    }

    with (
        patch("gpt_trader.monitoring.daily_report.generator.load_metrics") as mock_metrics,
        patch("gpt_trader.monitoring.daily_report.generator.load_events_since") as mock_events,
        patch(
            "gpt_trader.monitoring.daily_report.generator.load_liveness_snapshot",
            return_value=None,
        ) as mock_liveness,
        patch(
            "gpt_trader.monitoring.daily_report.generator.load_runtime_fingerprint",
            return_value=None,
        ) as mock_runtime,
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_pnl_metrics",
            return_value=pnl_metrics,
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_trade_metrics",
            return_value=trade_metrics,
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_symbol_metrics",
            return_value=[],
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_risk_metrics",
            return_value=risk_metrics,
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_health_metrics",
            return_value=health_metrics,
        ),
    ):
        mock_metrics.return_value = {"account": {"equity": 100}}
        mock_events.return_value = []

        report = generator.generate(date=date, lookback_hours=6)

    mock_metrics.assert_called_once_with(generator.metrics_file)
    mock_events.assert_called_once_with(generator.events_file, expected_cutoff)
    assert mock_liveness.call_count == 0
    assert mock_runtime.call_count == 0
    assert report.liveness is None
    assert report.runtime is None
    assert report.date == "2024-02-02"
    assert report.profile == "alpha"


def test_generate_liveness_uses_generated_at(tmp_path) -> None:
    date = datetime(2024, 2, 2, 12, 30, 0, tzinfo=UTC)
    generator = DailyReportGenerator(profile="alpha", data_dir=tmp_path)
    events_db = tmp_path / "events.db"
    events_db.write_text("db")

    with (
        patch("gpt_trader.monitoring.daily_report.generator.load_metrics", return_value={}),
        patch("gpt_trader.monitoring.daily_report.generator.load_events_since", return_value=[]),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_pnl_metrics",
            return_value={
                "equity": 100.0,
                "equity_change": 0.0,
                "equity_change_pct": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "funding_pnl": 0.0,
                "total_pnl": 0.0,
                "fees_paid": 0.0,
            },
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_trade_metrics",
            return_value={
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
            },
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_symbol_metrics", return_value=[]
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_risk_metrics",
            return_value={
                "guard_triggers": {},
                "circuit_breaker_state": {},
            },
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.calculate_health_metrics",
            return_value={
                "stale_marks_count": 0,
                "ws_reconnects": 0,
                "unfilled_orders": 0,
                "api_errors": 0,
            },
        ),
        patch(
            "gpt_trader.monitoring.daily_report.generator.load_liveness_snapshot"
        ) as mock_liveness,
        patch(
            "gpt_trader.monitoring.daily_report.generator.load_runtime_fingerprint",
            return_value=None,
        ),
        patch("gpt_trader.monitoring.daily_report.generator.datetime") as mock_datetime,
    ):
        generated_at = datetime(2024, 2, 2, 14, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = generated_at
        mock_datetime.UTC = UTC
        mock_datetime.side_effect = datetime

        report = generator.generate(date=date, lookback_hours=6)

    mock_liveness.assert_called_once_with(events_db, now=generated_at)
    assert report.generated_at == generated_at.isoformat()


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
