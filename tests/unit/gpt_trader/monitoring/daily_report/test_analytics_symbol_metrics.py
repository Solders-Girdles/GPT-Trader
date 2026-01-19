"""Tests for daily report symbol analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import calculate_symbol_metrics


class TestCalculateSymbolMetrics:
    """Tests for calculate_symbol_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_symbol_metrics([])
        assert result == []

    def test_skips_events_without_symbol(self) -> None:
        events = [{"type": "fill", "pnl": 100}]
        result = calculate_symbol_metrics(events)
        assert result == []

    def test_groups_by_symbol(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "ETH-USD", "pnl": 50},
        ]
        result = calculate_symbol_metrics(events)
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"BTC-USD", "ETH-USD"}

    def test_calculates_symbol_realized_pnl(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "BTC-USD", "pnl": 50},
        ]
        result = calculate_symbol_metrics(events)
        assert len(result) == 1
        assert result[0].realized_pnl == 150

    def test_tracks_unrealized_pnl(self) -> None:
        events = [
            {"type": "pnl_update", "symbol": "BTC-USD", "unrealized_pnl": 200},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].unrealized_pnl == 200

    def test_accumulates_funding_pnl(self) -> None:
        events = [
            {"type": "funding_payment", "symbol": "BTC-USD", "amount": 10},
            {"type": "funding_payment", "symbol": "BTC-USD", "amount": 5},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].funding_pnl == 15

    def test_calculates_total_pnl(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "pnl_update", "symbol": "BTC-USD", "unrealized_pnl": 50},
            {"type": "funding_payment", "symbol": "BTC-USD", "amount": 10},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].total_pnl == 160

    def test_counts_trades(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "BTC-USD", "pnl": -50},
            {"type": "fill", "symbol": "BTC-USD", "pnl": 30},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].trades == 3

    def test_calculates_win_rate(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "BTC-USD", "pnl": -50},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].win_rate == 0.5

    def test_calculates_avg_win_loss(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "BTC-USD", "pnl": 200},
            {"type": "fill", "symbol": "BTC-USD", "pnl": -60},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].avg_win == 150.0
        assert result[0].avg_loss == 60.0

    def test_calculates_profit_factor(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100},
            {"type": "fill", "symbol": "BTC-USD", "pnl": -50},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].profit_factor == 2.0

    def test_tracks_regime(self) -> None:
        events = [
            {"type": "fill", "symbol": "BTC-USD", "pnl": 100, "regime": "trending"},
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].regime == "trending"

    def test_tracks_exposure_from_position_update(self) -> None:
        events = [
            {
                "type": "position_update",
                "symbol": "BTC-USD",
                "quantity": 0.5,
                "price": 50000,
            },
        ]
        result = calculate_symbol_metrics(events)
        assert result[0].exposure_usd == 25000
