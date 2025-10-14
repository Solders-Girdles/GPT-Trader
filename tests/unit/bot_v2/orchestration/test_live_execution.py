"""Unit tests for LiveExecutionEngine.

Focus on pre-trade validation path and reduce-only enforcement.
"""

from decimal import Decimal
from datetime import datetime
from pathlib import Path
import tempfile
from typing import List, Optional

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    Balance,
    Product,
    MarketType,
    OrderSide,
    OrderType,
    Order,
    OrderStatus,
    TimeInForce,
    Position,
)
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore
from tests.support.deterministic_broker import DeterministicBroker


class DummyBroker:
    def __init__(self):
        self._product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
        )
        self._orders = {}

    # IBrokerage minimal surface
    def get_product(self, symbol: str) -> Product:  # type: ignore[override]
        return self._product

    def list_balances(self) -> list[Balance]:  # type: ignore[override]
        return [
            Balance(
                asset="USD", total=Decimal("1000"), available=Decimal("1000"), hold=Decimal("0")
            )
        ]

    def list_positions(self) -> list[Position]:  # type: ignore[override]
        return []

    def get_quote(self, symbol: str):  # type: ignore[override]
        # Minimal quote for price estimation when price is None
        from bot_v2.features.brokerages.core.interfaces import Quote

        return Quote(
            symbol=symbol,
            bid=Decimal("9999"),
            ask=Decimal("10001"),
            last=Decimal("10000"),
            ts=datetime.utcnow(),
        )

    def place_order(
        self,
        *,
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
    ) -> Order:  # type: ignore[override]
        oid = f"ord-{len(self._orders)+1}"
        order = Order(
            id=oid,
            client_id=client_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        return self._orders.pop(order_id, None) is not None


def test_place_order_success_with_explicit_reduce_only(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()  # default reduce_only_mode=False (allow increasing positions)

    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    # Patch validator inside module to always ok
    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=None,
        leverage=2,
        reduce_only=True,
    )

    assert isinstance(order_id, str)
    assert order_id in engine.open_orders


def test_reduce_only_mode_rejects_increasing_position(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()
    risk.set_reduce_only_mode(True)
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    with pytest.raises(ValidationError):
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            price=None,
            leverage=2,
        )


def test_place_order_rejected_raises_validation_error(monkeypatch):
    broker = DummyBroker()
    engine = LiveExecutionEngine(broker=broker)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    # Force spec validator to reject
    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(ok=False, reason="min_size"),
    )

    with pytest.raises(ValidationError):
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.0001"),
            price=Decimal("100"),
        )


def test_preview_enabled_calls_preview(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()
    risk.config.max_position_pct_per_symbol = 1.0
    risk.config.max_exposure_pct = 1.0
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk, enable_preview=True)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    preview_calls = []
    broker.preview_order = lambda **kwargs: preview_calls.append(kwargs) or {"success": True}  # type: ignore[attr-defined]
    broker.edit_order_preview = lambda order_id, **kwargs: None  # type: ignore[attr-defined]

    captured_metrics: list = []
    engine.event_store.append_metric = (
        lambda bot_id, metrics=None, **kwargs: captured_metrics.append((bot_id, metrics))
    )  # type: ignore

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
    )

    assert order_id is not None
    assert preview_calls
    assert any(m[1].get("event_type") == "order_preview" for m in captured_metrics)


def test_place_order_uses_usdc_collateral(monkeypatch):
    broker = DummyBroker()

    def usdc_balances() -> list[Balance]:
        return [
            Balance(
                asset="USDC", total=Decimal("5000"), available=Decimal("5000"), hold=Decimal("0")
            )
        ]

    broker.list_balances = usdc_balances  # type: ignore[assignment]

    risk = LiveRiskManager()
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=None,
    )

    assert isinstance(order_id, str)
    assert engine._last_collateral_available == Decimal("5000")


# ===== Perps-Specific Integration Tests =====


@pytest.mark.perps
def test_reduce_only_is_forced_in_execution_perps():
    """Test reduce_only flag propagates correctly in perps trading."""
    broker = DeterministicBroker()
    risk = LiveRiskManager(config=RiskConfig(reduce_only_mode=True))
    eng = LiveExecutionEngine(broker=broker, risk_manager=risk, event_store=EventStore())

    # Seed an existing long position so a SELL reduces
    broker.seed_position("BTC-PERP", side="long", quantity=Decimal("0.01"), price=Decimal("50000"))

    # Monkeypatch broker.place_order to capture reduce_only flag
    captured = {}
    orig = broker.place_order

    def _patched_place_order(**kwargs):  # type: ignore[no-redef]
        captured.update(kwargs)
        return orig(**kwargs)

    broker.place_order = _patched_place_order  # type: ignore[assignment]

    broker.connect()
    order_id = eng.place_order(
        symbol="BTC-PERP",
        side=OrderSide.SELL,  # reducing long
        order_type=OrderType.MARKET,
        quantity=Decimal("0.005"),
    )
    assert order_id is not None
    assert captured.get("reduce_only") is True


@pytest.mark.perps
def test_rejection_logs_metric_with_reason_perps():
    """Test order rejection logs proper metrics for perps orders."""
    broker = DeterministicBroker()
    tmp = tempfile.TemporaryDirectory()
    ev = EventStore(root=Path(tmp.name))
    risk = LiveRiskManager(config=RiskConfig(max_leverage=5, slippage_guard_bps=1_000_000))
    eng = LiveExecutionEngine(
        broker=broker, risk_manager=risk, event_store=ev, bot_id="perps_test_bot"
    )

    broker.connect()
    # Force huge quantity → leverage rejection
    with pytest.raises(ValidationError):
        eng.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("20"),  # notional >> equity
        )

    # Tail events and ensure order_rejected logged
    events = ev.tail("perps_test_bot", limit=20, types=["order_rejected"])  # type: ignore[arg-type]
    assert any("order_rejected" == e.get("type") for e in events)
    assert any("Leverage" in e.get("reason", "") for e in events)
