"""Tests for daily report DailyReport model fields."""

from .models_test_base import _create_daily_report


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
