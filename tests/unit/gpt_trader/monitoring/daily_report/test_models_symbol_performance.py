"""Tests for daily report SymbolPerformance model."""

from gpt_trader.monitoring.daily_report.models import SymbolPerformance

from .models_test_base import _create_symbol_performance


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
