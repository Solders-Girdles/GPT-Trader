"""Tests for daily report DailyReport.to_text()."""

from gpt_trader.monitoring.daily_report.models import SymbolPerformance

from .models_test_base import _create_daily_report, _create_symbol_performance


class TestDailyReportToText:
    """Tests for DailyReport.to_text method."""

    def test_returns_string(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert isinstance(result, str)

    def test_contains_header(self) -> None:
        report = _create_daily_report()
        report.liveness = {
            "status": "GREEN",
            "events": {
                "heartbeat": {"age_seconds": 10},
                "price_tick": {"age_seconds": 20},
            },
        }
        report.runtime = {"build_sha": "abc123"}
        result = report.to_text()
        assert "Daily Trading Report - 2024-01-15" in result
        assert "Profile: PROD" in result
        assert "RUNTIME" in result
        assert "Liveness:" in result
        assert "Build SHA" in result

    def test_contains_account_summary(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "ACCOUNT SUMMARY" in result
        assert "Equity:" in result

    def test_account_summary_handles_missing_equity(self) -> None:
        report = _create_daily_report(equity=None, equity_change_pct=None)
        result = report.to_text()
        assert "Equity:          N/A" in result
        assert "Change (24h):    $+500.00 (N/A)" in result

    def test_contains_pnl_breakdown(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "PnL BREAKDOWN" in result
        assert "Realized PnL:" in result
        assert "Unrealized PnL:" in result
        assert "Funding PnL:" in result

    def test_contains_performance_metrics(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "PERFORMANCE METRICS" in result
        assert "Win Rate:" in result
        assert "Profit Factor:" in result
        assert "Sharpe Ratio:" in result

    def test_contains_trade_statistics(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "TRADE STATISTICS" in result
        assert "Total Trades:" in result
        assert "Winning:" in result
        assert "Losing:" in result

    def test_contains_health_metrics(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "HEALTH METRICS" in result
        assert "Stale Marks:" in result
        assert "WS Reconnects:" in result

    def test_shows_no_guard_triggers(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "No guard triggers" in result

    def test_shows_guard_triggers(self) -> None:
        report = _create_daily_report(guard_triggers={"daily_loss": 2, "error_rate": 1})
        result = report.to_text()
        assert "Guard Triggers:" in result
        assert "daily_loss: 2" in result
        assert "error_rate: 1" in result

    def test_shows_circuit_breaker_ok(self) -> None:
        report = _create_daily_report()
        result = report.to_text()
        assert "Circuit breakers: OK" in result

    def test_shows_circuit_breaker_state(self) -> None:
        report = _create_daily_report(
            circuit_breaker_state={"trading_halt": True, "reason": "daily_loss"}
        )
        result = report.to_text()
        assert "Circuit Breaker State:" in result
        assert "trading_halt:" in result

    def test_contains_symbol_performance(self) -> None:
        symbols = [
            _create_symbol_performance(symbol="BTC-USD", total_pnl=200.0),
            _create_symbol_performance(symbol="ETH-USD", total_pnl=100.0),
        ]
        report = _create_daily_report(symbol_performance=symbols)
        result = report.to_text()
        assert "TOP PERFORMERS BY SYMBOL" in result
        assert "BTC-USD" in result
        assert "ETH-USD" in result

    def test_no_symbol_section_when_empty(self) -> None:
        report = _create_daily_report(symbol_performance=[])
        result = report.to_text()
        assert "TOP PERFORMERS BY SYMBOL" not in result

    def test_symbol_with_no_regime(self) -> None:
        symbols = [SymbolPerformance(symbol="BTC-USD", regime=None, total_pnl=100.0)]
        report = _create_daily_report(symbol_performance=symbols)
        result = report.to_text()
        assert "N/A" in result
