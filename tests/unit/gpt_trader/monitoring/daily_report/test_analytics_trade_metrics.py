"""Tests for daily report trade analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import calculate_trade_metrics


class TestCalculateTradeMetrics:
    """Tests for calculate_trade_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_trade_metrics([])
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0

    def test_counts_winning_trades(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": 50},
            {"type": "fill", "pnl": -30},
        ]
        result = calculate_trade_metrics(events)
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["total_trades"] == 3

    def test_calculates_win_rate(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": -50},
        ]
        result = calculate_trade_metrics(events)
        assert result["win_rate"] == 0.5

    def test_calculates_avg_win_loss(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": 200},
            {"type": "fill", "pnl": -50},
        ]
        result = calculate_trade_metrics(events)
        assert result["avg_win"] == 150.0
        assert result["avg_loss"] == 50.0

    def test_calculates_largest_win_loss(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": 200},
            {"type": "fill", "pnl": -50},
            {"type": "fill", "pnl": -150},
        ]
        result = calculate_trade_metrics(events)
        assert result["largest_win"] == 200
        assert result["largest_loss"] == 150

    def test_calculates_profit_factor(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": -50},
        ]
        result = calculate_trade_metrics(events)
        assert result["profit_factor"] == 2.0

    def test_profit_factor_zero_when_no_losses(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
        ]
        result = calculate_trade_metrics(events)
        assert result["profit_factor"] == 0

    def test_calculates_sharpe_ratio(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "fill", "pnl": 110},
            {"type": "fill", "pnl": 90},
        ]
        result = calculate_trade_metrics(events)
        assert result["sharpe_ratio"] > 0

    def test_sharpe_ratio_zero_with_single_trade(self) -> None:
        events = [{"type": "fill", "pnl": 100}]
        result = calculate_trade_metrics(events)
        assert result["sharpe_ratio"] == 0

    def test_calculates_max_drawdown(self) -> None:
        events = [
            {"type": "fill", "pnl": 100, "timestamp": "2024-01-01T01:00:00"},
            {"type": "fill", "pnl": -50, "timestamp": "2024-01-01T02:00:00"},
            {"type": "fill", "pnl": -30, "timestamp": "2024-01-01T03:00:00"},
        ]
        result = calculate_trade_metrics(events)
        assert result["max_drawdown"] == 80

    def test_calculates_max_drawdown_pct(self) -> None:
        events = [
            {"type": "fill", "pnl": 100, "timestamp": "2024-01-01T01:00:00"},
            {"type": "fill", "pnl": -50, "timestamp": "2024-01-01T02:00:00"},
        ]
        result = calculate_trade_metrics(events)
        assert result["max_drawdown_pct"] == 50.0

    def test_ignores_non_fill_events(self) -> None:
        events = [
            {"type": "fill", "pnl": 100},
            {"type": "position_update"},
            {"type": "pnl_update"},
        ]
        result = calculate_trade_metrics(events)
        assert result["total_trades"] == 1
