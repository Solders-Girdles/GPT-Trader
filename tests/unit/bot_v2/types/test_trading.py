"""Tests for trading domain types."""

from datetime import datetime
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.types.trading import (
    AccountSnapshot,
    OrderTicket,
    PerformanceSummary,
    TradeFill,
    TradingPosition,
    TradingSessionResult,
)


class TestTradingPosition:
    """Test the TradingPosition dataclass."""

    def test_trading_position_creation(self) -> None:
        """Test TradingPosition can be created with required fields."""
        position = TradingPosition(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
        )
        assert position.symbol == "BTC-USD"
        assert position.quantity == Decimal("1.5")
        assert position.entry_price == Decimal("50000.00")
        assert position.entry_timestamp is None
        assert position.current_price is None

    def test_trading_position_with_all_fields(self) -> None:
        """Test TradingPosition with all optional fields."""
        timestamp = datetime(2025, 10, 7, 12, 0, 0)
        position = TradingPosition(
            symbol="ETH-USD",
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000.00"),
            entry_timestamp=timestamp,
            current_price=Decimal("3100.00"),
            unrealized_pnl=Decimal("1000.00"),
            realized_pnl=Decimal("500.00"),
            value=Decimal("31000.00"),
        )
        assert position.symbol == "ETH-USD"
        assert position.entry_timestamp == timestamp
        assert position.current_price == Decimal("3100.00")
        assert position.unrealized_pnl == Decimal("1000.00")
        assert position.realized_pnl == Decimal("500.00")
        assert position.value == Decimal("31000.00")

    def test_trading_position_decimal_precision(self) -> None:
        """Test that TradingPosition maintains Decimal precision."""
        position = TradingPosition(
            symbol="BTC-USD",
            quantity=Decimal("0.00000001"),
            entry_price=Decimal("99999.99999999"),
        )
        assert isinstance(position.quantity, Decimal)
        assert isinstance(position.entry_price, Decimal)


class TestAccountSnapshot:
    """Test the AccountSnapshot dataclass."""

    def test_account_snapshot_creation(self) -> None:
        """Test AccountSnapshot can be created with required fields."""
        snapshot = AccountSnapshot(
            account_id="test-account-123",
            cash=Decimal("10000.00"),
            equity=Decimal("15000.00"),
            buying_power=Decimal("20000.00"),
            positions_value=Decimal("5000.00"),
            margin_used=Decimal("0.00"),
        )
        assert snapshot.account_id == "test-account-123"
        assert snapshot.cash == Decimal("10000.00")
        assert snapshot.equity == Decimal("15000.00")
        assert snapshot.buying_power == Decimal("20000.00")
        assert snapshot.positions_value == Decimal("5000.00")
        assert snapshot.margin_used == Decimal("0.00")

    def test_account_snapshot_with_optional_fields(self) -> None:
        """Test AccountSnapshot with pattern day trader fields."""
        snapshot = AccountSnapshot(
            account_id="pdt-account",
            cash=Decimal("25000.00"),
            equity=Decimal("30000.00"),
            buying_power=Decimal("120000.00"),
            positions_value=Decimal("5000.00"),
            margin_used=Decimal("5000.00"),
            pattern_day_trader=True,
            day_trades_remaining=2,
        )
        assert snapshot.pattern_day_trader is True
        assert snapshot.day_trades_remaining == 2

    def test_account_snapshot_none_account_id(self) -> None:
        """Test AccountSnapshot with None account_id (for simulated accounts)."""
        snapshot = AccountSnapshot(
            account_id=None,
            cash=Decimal("10000.00"),
            equity=Decimal("10000.00"),
            buying_power=Decimal("10000.00"),
            positions_value=Decimal("0.00"),
            margin_used=Decimal("0.00"),
        )
        assert snapshot.account_id is None


class TestTradeFill:
    """Test the TradeFill dataclass."""

    def test_trade_fill_creation(self) -> None:
        """Test TradeFill can be created with required fields."""
        timestamp = datetime(2025, 10, 7, 12, 0, 0)
        fill = TradeFill(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            timestamp=timestamp,
        )
        assert fill.symbol == "BTC-USD"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == Decimal("1.0")
        assert fill.price == Decimal("50000.00")
        assert fill.timestamp == timestamp
        assert fill.commission == Decimal("0")

    def test_trade_fill_with_string_side(self) -> None:
        """Test TradeFill accepts string literals for side."""
        timestamp = datetime(2025, 10, 7, 12, 0, 0)
        fill = TradeFill(
            symbol="ETH-USD",
            side="sell",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
            timestamp=timestamp,
        )
        assert fill.side == "sell"

    def test_trade_fill_with_all_fields(self) -> None:
        """Test TradeFill with all optional fields."""
        timestamp = datetime(2025, 10, 7, 12, 0, 0)
        fill = TradeFill(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            price=Decimal("51000.00"),
            timestamp=timestamp,
            commission=Decimal("25.50"),
            slippage=Decimal("50.00"),
            order_id="order-123",
            execution_id="exec-456",
        )
        assert fill.commission == Decimal("25.50")
        assert fill.slippage == Decimal("50.00")
        assert fill.order_id == "order-123"
        assert fill.execution_id == "exec-456"


class TestPerformanceSummary:
    """Test the PerformanceSummary dataclass."""

    def test_performance_summary_creation(self) -> None:
        """Test PerformanceSummary can be created with required fields."""
        summary = PerformanceSummary(
            total_return=0.15,
            max_drawdown=-0.08,
        )
        assert summary.total_return == 0.15
        assert summary.max_drawdown == -0.08
        assert summary.sharpe_ratio is None
        assert summary.win_rate is None

    def test_performance_summary_with_all_fields(self) -> None:
        """Test PerformanceSummary with all optional fields."""
        summary = PerformanceSummary(
            total_return=0.25,
            max_drawdown=-0.12,
            sharpe_ratio=1.8,
            win_rate=0.65,
            profit_factor=2.3,
            trades_count=150,
            daily_return=0.002,
        )
        assert summary.total_return == 0.25
        assert summary.max_drawdown == -0.12
        assert summary.sharpe_ratio == 1.8
        assert summary.win_rate == 0.65
        assert summary.profit_factor == 2.3
        assert summary.trades_count == 150
        assert summary.daily_return == 0.002

    def test_performance_summary_negative_returns(self) -> None:
        """Test PerformanceSummary handles negative performance."""
        summary = PerformanceSummary(
            total_return=-0.20,
            max_drawdown=-0.35,
            sharpe_ratio=-0.5,
            win_rate=0.40,
            profit_factor=0.7,
        )
        assert summary.total_return == -0.20
        assert summary.sharpe_ratio == -0.5
        assert summary.profit_factor == 0.7


class TestTradingSessionResult:
    """Test the TradingSessionResult dataclass."""

    def test_trading_session_result_creation(self) -> None:
        """Test TradingSessionResult can be created with required fields."""
        start = datetime(2025, 10, 7, 9, 0, 0)
        account = AccountSnapshot(
            account_id="test-123",
            cash=Decimal("10000.00"),
            equity=Decimal("10000.00"),
            buying_power=Decimal("10000.00"),
            positions_value=Decimal("0.00"),
            margin_used=Decimal("0.00"),
        )
        result = TradingSessionResult(
            start_time=start,
            end_time=None,
            account=account,
        )
        assert result.start_time == start
        assert result.end_time is None
        assert result.account == account
        assert result.positions == []
        assert result.fills == []
        assert result.performance is None

    def test_trading_session_result_with_all_fields(self) -> None:
        """Test TradingSessionResult with all optional fields."""
        start = datetime(2025, 10, 7, 9, 0, 0)
        end = datetime(2025, 10, 7, 16, 0, 0)
        account = AccountSnapshot(
            account_id="test-123",
            cash=Decimal("12000.00"),
            equity=Decimal("15000.00"),
            buying_power=Decimal("15000.00"),
            positions_value=Decimal("3000.00"),
            margin_used=Decimal("0.00"),
        )
        position = TradingPosition(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )
        fill = TradeFill(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            timestamp=start,
        )
        performance = PerformanceSummary(
            total_return=0.20,
            max_drawdown=-0.05,
        )
        result = TradingSessionResult(
            start_time=start,
            end_time=end,
            account=account,
            positions=[position],
            fills=[fill],
            performance=performance,
        )
        assert result.end_time == end
        assert len(result.positions) == 1
        assert result.positions[0] == position
        assert len(result.fills) == 1
        assert result.fills[0] == fill
        assert result.performance == performance


class TestOrderTicket:
    """Test the OrderTicket dataclass."""

    def test_order_ticket_creation(self) -> None:
        """Test OrderTicket can be created with required fields."""
        ticket = OrderTicket(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        assert ticket.symbol == "BTC-USD"
        assert ticket.side == OrderSide.BUY
        assert ticket.order_type == OrderType.MARKET
        assert ticket.quantity == Decimal("1.0")
        assert ticket.price is None
        assert ticket.stop_price is None

    def test_order_ticket_limit_order(self) -> None:
        """Test OrderTicket for limit order with price."""
        ticket = OrderTicket(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("3100.00"),
            time_in_force="gtc",
        )
        assert ticket.order_type == OrderType.LIMIT
        assert ticket.price == Decimal("3100.00")
        assert ticket.time_in_force == "gtc"

    def test_order_ticket_stop_order(self) -> None:
        """Test OrderTicket for stop order with stop_price."""
        ticket = OrderTicket(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("0.5"),
            stop_price=Decimal("48000.00"),
            time_in_force="ioc",
        )
        assert ticket.stop_price == Decimal("48000.00")
        assert ticket.time_in_force == "ioc"

    def test_order_ticket_with_client_id(self) -> None:
        """Test OrderTicket with custom client_id."""
        ticket = OrderTicket(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            client_id="custom-order-123",
        )
        assert ticket.client_id == "custom-order-123"

    def test_order_ticket_all_time_in_force_options(self) -> None:
        """Test OrderTicket with all time_in_force options."""
        for tif in ["gtc", "ioc", "fok"]:
            ticket = OrderTicket(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                time_in_force=tif,  # type: ignore
            )
            assert ticket.time_in_force == tif


class TestModuleExports:
    """Test that all expected types are exported."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains all expected exports."""
        from bot_v2.types import trading

        expected_exports = [
            "TradingPosition",
            "AccountSnapshot",
            "TradeFill",
            "PerformanceSummary",
            "TradingSessionResult",
            "OrderTicket",
        ]
        assert set(trading.__all__) == set(expected_exports)

    def test_all_types_importable(self) -> None:
        """Test that all types can be imported."""
        # This test passes if the imports at the top of the file succeed
        assert TradingPosition is not None
        assert AccountSnapshot is not None
        assert TradeFill is not None
        assert PerformanceSummary is not None
        assert TradingSessionResult is not None
        assert OrderTicket is not None
