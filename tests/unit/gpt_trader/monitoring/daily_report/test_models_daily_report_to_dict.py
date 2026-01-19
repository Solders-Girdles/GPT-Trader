"""Tests for daily report DailyReport.to_dict()."""

from .models_test_base import _create_daily_report, _create_symbol_performance


class TestDailyReportToDict:
    """Tests for DailyReport.to_dict method."""

    def test_returns_dict(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_date_and_profile(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["date"] == "2024-01-15"
        assert result["profile"] == "PROD"
        assert result["generated_at"] == "2024-01-15T12:00:00"

    def test_account_section(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["account"]["equity"] == 100000.0
        assert result["account"]["equity_change"] == 500.0
        assert result["account"]["equity_change_pct"] == 0.5

    def test_account_section_with_missing_equity(self) -> None:
        report = _create_daily_report(equity=None, equity_change_pct=None)
        result = report.to_dict()
        assert result["account"]["equity"] is None
        assert result["account"]["equity_change"] == 500.0
        assert result["account"]["equity_change_pct"] is None

    def test_pnl_section(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["pnl"]["realized"] == 400.0
        assert result["pnl"]["unrealized"] == 100.0
        assert result["pnl"]["funding"] == -25.0
        assert result["pnl"]["total"] == 475.0
        assert result["pnl"]["fees"] == 50.0

    def test_performance_section(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["performance"]["win_rate"] == 0.65
        assert result["performance"]["profit_factor"] == 2.5
        assert result["performance"]["sharpe_ratio"] == 1.8
        assert result["performance"]["max_drawdown"] == 5000.0
        assert result["performance"]["max_drawdown_pct"] == 5.0

    def test_trades_section(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["trades"]["total"] == 20
        assert result["trades"]["winning"] == 13
        assert result["trades"]["losing"] == 7
        assert result["trades"]["avg_win"] == 100.0
        assert result["trades"]["avg_loss"] == -50.0
        assert result["trades"]["largest_win"] == 500.0
        assert result["trades"]["largest_loss"] == -200.0

    def test_risk_section(self) -> None:
        report = _create_daily_report(
            guard_triggers={"daily_loss": 2},
            circuit_breaker_state={"enabled": True},
        )
        result = report.to_dict()
        assert result["risk"]["guard_triggers"] == {"daily_loss": 2}
        assert result["risk"]["circuit_breaker_state"] == {"enabled": True}

    def test_symbols_section(self) -> None:
        symbols = [
            _create_symbol_performance(symbol="BTC-USD", total_pnl=100.0),
            _create_symbol_performance(symbol="ETH-USD", total_pnl=50.0),
        ]
        report = _create_daily_report(symbol_performance=symbols)
        result = report.to_dict()
        assert len(result["symbols"]) == 2
        assert result["symbols"][0]["symbol"] == "BTC-USD"
        assert result["symbols"][1]["symbol"] == "ETH-USD"

    def test_symbol_pnl_nested(self) -> None:
        symbols = [
            _create_symbol_performance(
                symbol="BTC-USD",
                realized_pnl=100.0,
                unrealized_pnl=50.0,
                funding_pnl=-5.0,
                total_pnl=145.0,
            ),
        ]
        report = _create_daily_report(symbol_performance=symbols)
        result = report.to_dict()
        assert result["symbols"][0]["pnl"]["realized"] == 100.0
        assert result["symbols"][0]["pnl"]["unrealized"] == 50.0
        assert result["symbols"][0]["pnl"]["funding"] == -5.0
        assert result["symbols"][0]["pnl"]["total"] == 145.0

    def test_health_section(self) -> None:
        report = _create_daily_report()
        result = report.to_dict()
        assert result["health"]["stale_marks"] == 2
        assert result["health"]["ws_reconnects"] == 1
        assert result["health"]["unfilled_orders"] == 0
        assert result["health"]["api_errors"] == 3

    def test_health_liveness_and_runtime_fields(self) -> None:
        report = _create_daily_report()
        report.liveness = {"status": "GREEN"}
        report.runtime = {"event_id": 123}
        result = report.to_dict()

        assert result["health"]["liveness"] == {"status": "GREEN"}
        assert result["runtime"] == {"event_id": 123}
