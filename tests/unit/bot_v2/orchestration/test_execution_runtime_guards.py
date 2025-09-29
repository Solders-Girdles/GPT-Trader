from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskConfig
from bot_v2.features.live_trade import guard_errors
from bot_v2.features.live_trade.guard_errors import (
    RiskGuardTelemetryError,
    RiskGuardComputationError,
)
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction, CircuitBreakerOutcome
from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    Product,
    MarketType,
    Quote,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    Balance,
    Position,
)
from bot_v2.monitoring.metrics_collector import get_metrics_collector


class SimpleBroker(IBrokerage):
    def __init__(self):
        self._orders = {}

    def list_balances(self) -> list[Balance]:
        return [
            Balance(
                asset="USD", total=Decimal("100000"), available=Decimal("100000"), hold=Decimal("0")
            )
        ]

    def get_product(self, symbol: str) -> Product:
        return Product(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("100.10"),
            last=Decimal("100.05"),
            ts=datetime.utcnow(),
        )

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
    ) -> Order:
        now = datetime.utcnow()
        oid = client_id or f"ord_{len(self._orders)+1}"
        o = Order(
            id=oid,
            client_id=oid,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=price or Decimal("100.10") if side == OrderSide.BUY else Decimal("100"),
            submitted_at=now,
            updated_at=now,
        )
        self._orders[oid] = o
        return o

    # Minimal protocol impl
    def list_positions(self) -> list[Position]:
        return []

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def list_orders(self, status: OrderStatus | None = None, symbol: str | None = None):
        return list(self._orders.values())

    def list_products(self, market: MarketType | None = None):  # pragma: no cover
        return []

    def connect(self) -> bool:  # pragma: no cover
        return True

    def disconnect(self) -> None:  # pragma: no cover
        pass

    def validate_connection(self) -> bool:  # pragma: no cover
        return True

    def list_fills(self, symbol: str | None = None, limit: int = 200):  # pragma: no cover
        return []

    def stream_trades(self, symbols):  # pragma: no cover
        return []

    def stream_orderbook(self, symbols, level: int = 1):  # pragma: no cover
        return []


class CountingBroker(SimpleBroker):
    def __init__(self):
        super().__init__()
        self.balance_calls = 0
        self.position_calls = 0

    def list_balances(self) -> list[Balance]:
        self.balance_calls += 1
        return super().list_balances()

    def list_positions(self) -> list[Position]:
        self.position_calls += 1
        return []


def test_slippage_multiplier_path_and_cancel_all_orders():
    broker = SimpleBroker()
    risk_config = RiskConfig(
        slippage_guard_bps=10000, max_position_pct_per_symbol=0.9, max_exposure_pct=0.9
    )  # disable slippage guard
    risk = LiveRiskManager(config=risk_config)
    engine = LiveExecutionEngine(
        broker,
        risk_manager=risk,
        event_store=None,
        bot_id="test",
        slippage_multipliers={"BTC-PERP": 0.001},
    )

    oid = engine.place_order(
        symbol="BTC-PERP", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.2")
    )
    assert oid is not None
    cancelled = engine.cancel_all_orders()
    assert cancelled == 1
    assert engine.open_orders == []


def test_runtime_guards_use_cached_state() -> None:
    broker = CountingBroker()
    risk = LiveRiskManager(config=RiskConfig())
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    engine.run_runtime_guards()
    assert broker.balance_calls == 1
    assert broker.position_calls == 1

    engine.run_runtime_guards()
    assert broker.balance_calls == 1
    assert broker.position_calls == 1

    engine._invalidate_runtime_guard_cache()
    engine.run_runtime_guards()
    assert broker.balance_calls == 2
    assert broker.position_calls == 2


class TelemetryFailRisk(LiveRiskManager):
    """Risk manager that simulates telemetry failure during daily loss guard."""

    def __init__(self):
        super().__init__(config=RiskConfig())

    def track_daily_pnl(self, current_equity, positions_pnl):  # type: ignore[override]
        raise RiskGuardTelemetryError(guard="daily_loss", message="telemetry failure")

    def check_liquidation_buffer(self, symbol, position_data, equity):  # type: ignore[override]
        return False

    def append_risk_metrics(self, equity, positions):  # type: ignore[override]
        return None

    def check_correlation_risk(self, positions):  # type: ignore[override]
        return False

    def check_volatility_circuit_breaker(self, symbol, recent_marks):  # type: ignore[override]
        return CircuitBreakerOutcome(
            guard="volatility_circuit_breaker",
            symbol=symbol,
            action=CircuitBreakerAction.NONE,
            triggered=False,
            triggered_at=None,
            cooldown_expires=None,
            telemetry={},
            metric=None,
        )


class FakeAlertSystem:
    def __init__(self):
        self.alerts = []

    def trigger_alert(self, level, category, message, metadata=None):
        self.alerts.append((level, category, message, metadata))


def test_runtime_guard_recoverable_failure_records_counter():
    metrics = get_metrics_collector()
    metrics.reset_all()

    broker = SimpleBroker()
    engine = LiveExecutionEngine(broker=broker, risk_manager=TelemetryFailRisk())

    fake_alerts = FakeAlertSystem()
    guard_errors.configure_guard_alert_system(fake_alerts)

    try:
        engine.run_runtime_guards()
    finally:
        guard_errors.configure_guard_alert_system(None)

    summary = metrics.get_metrics_summary()
    counters = summary["counters"]
    assert counters.get("risk.guards.daily_loss.recoverable_failures") == 1
    assert counters.get("risk.guards.daily_loss.telemetry") == 1
    metrics.reset_all()
    assert len(fake_alerts.alerts) == 1


class CriticalFailRisk(LiveRiskManager):
    """Risk manager that simulates a fatal failure in liquidation buffer guard."""

    def __init__(self):
        super().__init__(config=RiskConfig())

    def track_daily_pnl(self, current_equity, positions_pnl):  # type: ignore[override]
        return False

    def check_liquidation_buffer(self, symbol, position_data, equity):  # type: ignore[override]
        raise RiskGuardComputationError(guard="liquidation_buffer", message="math failure")

    def append_risk_metrics(self, equity, positions):  # type: ignore[override]
        return None

    def check_correlation_risk(self, positions):  # type: ignore[override]
        return False

    def check_volatility_circuit_breaker(self, symbol, recent_marks):  # type: ignore[override]
        return CircuitBreakerOutcome(
            guard="volatility_circuit_breaker",
            symbol=symbol,
            action=CircuitBreakerAction.NONE,
            triggered=False,
            triggered_at=None,
            cooldown_expires=None,
            telemetry={},
            metric=None,
        )


def test_runtime_guard_critical_failure_sets_reduce_only():
    metrics = get_metrics_collector()
    metrics.reset_all()

    class PositionBroker(SimpleBroker):
        def list_positions(self) -> list[Position]:
            return [
                Position(
                    symbol="BTC-PERP",
                    quantity=Decimal("1"),
                    side="long",
                    entry_price=Decimal("10000"),
                    mark_price=Decimal("10100"),
                    unrealized_pnl=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    leverage=1,
                )
            ]

    broker = PositionBroker()
    risk = CriticalFailRisk()
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    fake_alerts = FakeAlertSystem()
    guard_errors.configure_guard_alert_system(fake_alerts)

    try:
        engine.run_runtime_guards()
    finally:
        guard_errors.configure_guard_alert_system(None)

    assert risk.is_reduce_only_mode() is True
    summary = metrics.get_metrics_summary()
    counters = summary["counters"]
    assert counters.get("risk.guards.liquidation_buffer.critical_failures") == 1
    assert counters.get("risk.guards.liquidation_buffer.computation") == 1
    metrics.reset_all()
    assert len(fake_alerts.alerts) == 1
