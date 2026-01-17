"""Tests for daily report models."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.models import DailyReport, SymbolPerformance


def _create_symbol_performance(
    symbol: str = "BTC-USD",
    regime: str | None = "trending",
    realized_pnl: float = 100.0,
    unrealized_pnl: float = 50.0,
    funding_pnl: float = -5.0,
    total_pnl: float = 145.0,
    trades: int = 10,
    win_rate: float = 0.6,
) -> SymbolPerformance:
    """Create a test SymbolPerformance instance."""
    return SymbolPerformance(
        symbol=symbol,
        regime=regime,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        funding_pnl=funding_pnl,
        total_pnl=total_pnl,
        trades=trades,
        win_rate=win_rate,
    )


def _create_daily_report(
    symbol_performance: list[SymbolPerformance] | None = None,
    guard_triggers: dict[str, int] | None = None,
    circuit_breaker_state: dict | None = None,
    equity: float | None = 100000.0,
    equity_change_pct: float | None = 0.5,
) -> DailyReport:
    """Create a test DailyReport instance."""
    return DailyReport(
        date="2024-01-15",
        profile="PROD",
        generated_at="2024-01-15T12:00:00",
        equity=equity,
        equity_change=500.0,
        equity_change_pct=equity_change_pct,
        realized_pnl=400.0,
        unrealized_pnl=100.0,
        funding_pnl=-25.0,
        total_pnl=475.0,
        fees_paid=50.0,
        win_rate=0.65,
        profit_factor=2.5,
        sharpe_ratio=1.8,
        max_drawdown=5000.0,
        max_drawdown_pct=5.0,
        total_trades=20,
        winning_trades=13,
        losing_trades=7,
        avg_win=100.0,
        avg_loss=-50.0,
        largest_win=500.0,
        largest_loss=-200.0,
        guard_triggers=guard_triggers or {},
        circuit_breaker_state=circuit_breaker_state or {},
        symbol_performance=symbol_performance or [],
        stale_marks_count=2,
        ws_reconnects=1,
        unfilled_orders=0,
        api_errors=3,
    )


class TestSymbolPerformance:
    """Tests for SymbolPerformance dataclass."""

    def test_creation_with_required_field(self) -> None:
        perf = SymbolPerformance(symbol="ETH-USD")
        assert perf.symbol == "ETH-USD"

    def test_defaults(self) -> None:
        perf = SymbolPerformance(symbol="BTC-USD")
        assert perf.regime is None
        assert perf.realized_pnl == 0.0
        assert perf.unrealized_pnl == 0.0
        assert perf.funding_pnl == 0.0
        assert perf.total_pnl == 0.0
        assert perf.trades == 0
        assert perf.win_rate == 0.0
        assert perf.avg_win == 0.0
        assert perf.avg_loss == 0.0
        assert perf.profit_factor == 0.0
        assert perf.exposure_usd == 0.0

    def test_full_creation(self) -> None:
        perf = _create_symbol_performance()
        assert perf.symbol == "BTC-USD"
        assert perf.regime == "trending"
        assert perf.realized_pnl == 100.0
        assert perf.unrealized_pnl == 50.0
        assert perf.total_pnl == 145.0


class TestDailyReport:
    """Tests for DailyReport dataclass."""

    def test_creation(self) -> None:
        report = _create_daily_report()
        assert report.date == "2024-01-15"
        assert report.profile == "PROD"
        assert report.equity == 100000.0

    def test_pnl_fields(self) -> None:
        report = _create_daily_report()
        assert report.realized_pnl == 400.0
        assert report.unrealized_pnl == 100.0
        assert report.funding_pnl == -25.0
        assert report.total_pnl == 475.0
        assert report.fees_paid == 50.0

    def test_performance_fields(self) -> None:
        report = _create_daily_report()
        assert report.win_rate == 0.65
        assert report.profit_factor == 2.5
        assert report.sharpe_ratio == 1.8

    def test_trade_fields(self) -> None:
        report = _create_daily_report()
        assert report.total_trades == 20
        assert report.winning_trades == 13
        assert report.losing_trades == 7

    def test_health_fields(self) -> None:
        report = _create_daily_report()
        assert report.stale_marks_count == 2
        assert report.ws_reconnects == 1
        assert report.unfilled_orders == 0
        assert report.api_errors == 3


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
