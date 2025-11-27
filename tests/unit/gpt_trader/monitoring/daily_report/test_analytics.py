"""Tests for daily report analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import (
    calculate_health_metrics,
    calculate_pnl_metrics,
    calculate_risk_metrics,
    calculate_symbol_metrics,
    calculate_trade_metrics,
)


class TestCalculatePnlMetrics:
    """Tests for calculate_pnl_metrics function."""

    def test_returns_zero_equity_when_missing(self) -> None:
        result = calculate_pnl_metrics([], {})
        assert result["equity"] == 0

    def test_extracts_equity_from_account(self) -> None:
        result = calculate_pnl_metrics(
            [],
            {"account": {"equity": 10000.0}},
        )
        assert result["equity"] == 10000.0

    def test_accumulates_realized_pnl_from_events(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0},
            {"type": "pnl_update", "realized_pnl": 50.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["realized_pnl"] == 150.0

    def test_uses_last_unrealized_pnl(self) -> None:
        events = [
            {"type": "pnl_update", "unrealized_pnl": 100.0},
            {"type": "pnl_update", "unrealized_pnl": 200.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["unrealized_pnl"] == 200.0

    def test_accumulates_funding_payments(self) -> None:
        events = [
            {"type": "funding_payment", "amount": 10.0},
            {"type": "funding_payment", "amount": -5.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["funding_pnl"] == 5.0

    def test_accumulates_fees_from_fills(self) -> None:
        events = [
            {"type": "fill", "fee": 1.5},
            {"type": "fill", "fee": 2.5},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["fees_paid"] == 4.0

    def test_calculates_total_pnl(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 50.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["total_pnl"] == 150.0

    def test_calculates_equity_change(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 1100.0}})
        assert result["equity_change"] == 100.0

    def test_calculates_equity_change_pct(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 1100.0}})
        assert result["equity_change_pct"] == 10.0

    def test_equity_change_pct_zero_when_no_prev_equity(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 100.0}})
        assert result["equity_change_pct"] == 0


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


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_risk_metrics([])
        assert result["guard_triggers"] == {}
        assert result["circuit_breaker_state"] == {}

    def test_counts_guard_triggers(self) -> None:
        events = [
            {"type": "guard_triggered", "guard": "daily_loss"},
            {"type": "guard_triggered", "guard": "daily_loss"},
            {"type": "guard_triggered", "guard": "volatility"},
        ]
        result = calculate_risk_metrics(events)
        assert result["guard_triggers"]["daily_loss"] == 2
        assert result["guard_triggers"]["volatility"] == 1

    def test_handles_unknown_guard(self) -> None:
        events = [{"type": "guard_triggered"}]
        result = calculate_risk_metrics(events)
        assert result["guard_triggers"]["unknown"] == 1

    def test_tracks_circuit_breaker_state(self) -> None:
        events = [
            {
                "type": "circuit_breaker_triggered",
                "rule": "max_drawdown",
                "action": "halt_trading",
                "timestamp": "2024-01-01T12:00:00",
            },
        ]
        result = calculate_risk_metrics(events)
        assert result["circuit_breaker_state"]["triggered"] is True
        assert result["circuit_breaker_state"]["rule"] == "max_drawdown"
        assert result["circuit_breaker_state"]["action"] == "halt_trading"


class TestCalculateHealthMetrics:
    """Tests for calculate_health_metrics function."""

    def test_empty_events(self) -> None:
        result = calculate_health_metrics([])
        assert result["stale_marks_count"] == 0
        assert result["ws_reconnects"] == 0
        assert result["unfilled_orders"] == 0
        assert result["api_errors"] == 0

    def test_counts_stale_marks(self) -> None:
        events = [
            {"type": "stale_mark_detected"},
            {"type": "stale_mark_detected"},
        ]
        result = calculate_health_metrics(events)
        assert result["stale_marks_count"] == 2

    def test_counts_ws_reconnects(self) -> None:
        events = [
            {"type": "websocket_reconnect"},
            {"type": "websocket_reconnect"},
            {"type": "websocket_reconnect"},
        ]
        result = calculate_health_metrics(events)
        assert result["ws_reconnects"] == 3

    def test_counts_unfilled_orders(self) -> None:
        events = [{"type": "unfilled_order_alert"}]
        result = calculate_health_metrics(events)
        assert result["unfilled_orders"] == 1

    def test_counts_api_errors(self) -> None:
        events = [
            {"type": "api_error"},
            {"type": "api_error"},
        ]
        result = calculate_health_metrics(events)
        assert result["api_errors"] == 2

    def test_counts_all_types(self) -> None:
        events = [
            {"type": "stale_mark_detected"},
            {"type": "websocket_reconnect"},
            {"type": "unfilled_order_alert"},
            {"type": "api_error"},
        ]
        result = calculate_health_metrics(events)
        assert result["stale_marks_count"] == 1
        assert result["ws_reconnects"] == 1
        assert result["unfilled_orders"] == 1
        assert result["api_errors"] == 1
