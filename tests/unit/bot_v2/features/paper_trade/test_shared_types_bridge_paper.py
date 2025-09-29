from datetime import datetime
from decimal import Decimal

import pandas as pd

from bot_v2.features.paper_trade.types import (
    AccountStatus,
    PaperTradeResult,
    PerformanceMetrics,
    Position,
    TradeLog,
)
from bot_v2.features.paper_trade.risk import RiskManager
from bot_v2.types.trading import AccountSnapshot


def test_position_conversion_roundtrip() -> None:
    position = Position(
        symbol="BTC",
        quantity=5,
        entry_price=10000.0,
        entry_date=datetime(2024, 1, 1),
        current_price=10250.0,
        unrealized_pnl=1250.0,
        value=51250.0,
    )

    shared = position.to_trading_position()
    assert shared.quantity == Decimal("5")
    assert shared.entry_price == Decimal("10000.0")
    assert shared.current_price == Decimal("10250.0")

    restored = Position.from_trading_position(shared)
    assert restored.symbol == position.symbol
    assert restored.quantity == position.quantity
    assert restored.entry_price == position.entry_price
    assert restored.value == position.value


def test_account_snapshot_conversion() -> None:
    account = AccountStatus(
        cash=25000.0,
        positions_value=75000.0,
        total_equity=100000.0,
        buying_power=50000.0,
        margin_used=20000.0,
        day_trades_remaining=3,
    )

    snapshot = account.to_account_snapshot(account_id="acct-1")
    assert snapshot.cash == Decimal("25000.0")
    assert snapshot.equity == Decimal("100000.0")

    restored = AccountStatus.from_account_snapshot(snapshot)
    assert restored.cash == account.cash
    assert restored.total_equity == account.total_equity
    assert restored.day_trades_remaining == account.day_trades_remaining


def test_trade_log_to_fill() -> None:
    log = TradeLog(
        id=42,
        symbol="ETH",
        side="buy",
        quantity=2,
        price=1500.0,
        timestamp=datetime(2024, 1, 2, 12, 0, 0),
        commission=3.0,
        slippage=1.5,
    )

    fill = log.to_trade_fill()
    assert fill.symbol == "ETH"
    assert fill.quantity == Decimal("2")
    assert fill.price == Decimal("1500.0")
    assert fill.order_id == "42"


def test_paper_trade_result_to_shared_session() -> None:
    account = AccountStatus(
        cash=1000.0,
        positions_value=2000.0,
        total_equity=3000.0,
        buying_power=1500.0,
        margin_used=500.0,
        day_trades_remaining=2,
    )
    position = Position(
        symbol="SOL",
        quantity=10,
        entry_price=20.0,
        entry_date=datetime(2024, 1, 3),
        current_price=22.0,
        unrealized_pnl=20.0,
        value=220.0,
    )
    trade = TradeLog(
        id=1,
        symbol="SOL",
        side="sell",
        quantity=5,
        price=21.5,
        timestamp=datetime(2024, 1, 3, 15, 0, 0),
        commission=1.0,
        slippage=0.5,
    )
    performance_metrics = PerformanceMetrics(
        total_return=0.05,
        daily_return=0.01,
        sharpe_ratio=1.2,
        max_drawdown=0.02,
        win_rate=0.6,
        profit_factor=1.5,
        trades_count=10,
    )

    result = PaperTradeResult(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 4),
        account_status=account,
        positions=[position],
        trade_log=[trade],
        performance=performance_metrics,
        equity_curve=pd.Series([3000.0]),
    )

    session = result.to_trading_session(account_id="acct-7")
    assert session.account.account_id == "acct-7"
    assert session.positions[0].symbol == "SOL"
    assert session.fills[0].side.value == "sell"


def test_risk_manager_accepts_account_snapshot() -> None:
    manager = RiskManager(
        max_position_size=1.0,
        max_daily_loss=1.0,
        max_drawdown=1.0,
        min_cash_reserve=0.0,
    )
    account = AccountStatus(
        cash=5000.0,
        positions_value=0.0,
        total_equity=5000.0,
        buying_power=10000.0,
        margin_used=0.0,
        day_trades_remaining=3,
    )
    snapshot = account.to_account_snapshot()

    assert manager.check_trade("BTC", 1, 50.0, snapshot)
    metrics = manager.get_risk_metrics(snapshot)
    assert "current_drawdown" in metrics
