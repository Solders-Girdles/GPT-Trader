from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import OrderSide, Position
from bot_v2.features.live_trade.types import (
    AccountInfo,
    ExecutionReport,
    position_to_trading_position,
)
from bot_v2.types.trading import AccountSnapshot, TradeFill


def test_account_info_conversion_roundtrip() -> None:
    account = AccountInfo(
        account_id="acct-123",
        cash=25000.0,
        portfolio_value=75000.0,
        buying_power=50000.0,
        positions_value=30000.0,
        margin_used=10000.0,
        pattern_day_trader=True,
        day_trades_remaining=2,
        equity=100000.0,
        last_equity=99000.0,
    )

    snapshot = account.to_account_snapshot()
    assert snapshot.account_id == "acct-123"
    assert snapshot.cash == Decimal("25000.0")
    assert snapshot.day_trades_remaining == 2

    restored = AccountInfo.from_account_snapshot(snapshot)
    assert restored.account_id == account.account_id
    assert restored.cash == account.cash
    assert restored.pattern_day_trader == account.pattern_day_trader


def test_execution_report_to_trade_fill() -> None:
    report = ExecutionReport(
        order_id="order-1",
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=3,
        price=1500.5,
        commission=1.25,
        timestamp=datetime(2024, 6, 1, 12, 0, 0),
        execution_id="exec-1",
    )

    fill = report.to_trade_fill()
    assert fill.symbol == "ETH-USD"
    assert fill.quantity == Decimal("3")
    assert fill.side is OrderSide.BUY
    assert fill.commission == Decimal("1.25")

    reconstructed = ExecutionReport.from_trade_fill(fill)
    assert reconstructed.order_id == "order-1"
    assert reconstructed.quantity == 3
    assert reconstructed.side is OrderSide.BUY


def test_position_to_trading_position() -> None:
    position = Position(
        symbol="BTC-USD",
        quantity=Decimal("0.5"),
        entry_price=Decimal("25000"),
        mark_price=Decimal("26000"),
        unrealized_pnl=Decimal("500"),
        realized_pnl=Decimal("100"),
        leverage=3,
        side="long",
    )

    trading_position = position_to_trading_position(position)
    assert trading_position.symbol == "BTC-USD"
    assert trading_position.quantity == Decimal("0.5")
    assert trading_position.current_price == Decimal("26000")
    assert trading_position.unrealized_pnl == Decimal("500")
