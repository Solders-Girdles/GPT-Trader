from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


def _build_context(
    *,
    broker: object,
    risk_manager: object,
    event_store: object,
    strategy_coordinator: object,
) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"])
    config.perps_enable_streaming = True
    config.perps_stream_level = 1
    config.short_ma = 2
    config.long_ma = 3

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
    )

    runtime_state = PerpsBotRuntimeState(config.symbols)

    return CoordinatorContext(
        config=config,
        registry=registry,
        event_store=registry.event_store,
        orders_store=None,
        broker=broker,
        risk_manager=risk_manager,
        symbols=tuple(config.symbols),
        bot_id="perps_bot",
        runtime_state=runtime_state,
        strategy_coordinator=strategy_coordinator,
        set_running_flag=lambda _: None,
    )


def test_stream_loop_updates_mark_window_and_metrics() -> None:
    updates: list[tuple[str, Decimal]] = []

    class StubStrategy:
        def update_mark_window(self, symbol: str, mark: Decimal) -> None:
            updates.append((symbol, mark))

    class StubRisk:
        def __init__(self) -> None:
            self.last_mark_update: dict[str, object] = {}

        def record_mark_update(self, symbol: str, ts: object) -> object:
            self.last_mark_update[symbol] = ts
            return ts

    class StubEventStore:
        def __init__(self) -> None:
            self.metrics: list[tuple[str, dict[str, object]]] = []

        def append_metric(self, bot_id: str, metrics: dict[str, object]) -> None:
            self.metrics.append((bot_id, dict(metrics)))

    class StubBroker:
        def stream_orderbook(self, symbols: list[str], *, level: int):
            yield {"product_id": symbols[0], "best_bid": "100", "best_ask": "102"}

    strategy = StubStrategy()
    risk = StubRisk()
    event_store = StubEventStore()
    broker = StubBroker()

    context = _build_context(
        broker=broker,
        risk_manager=risk,
        event_store=event_store,
        strategy_coordinator=strategy,
    )

    coordinator = TelemetryCoordinator(context)
    coordinator.update_context(context)
    coordinator._market_monitor = None  # type: ignore[attr-defined]

    coordinator._run_stream_loop(["BTC-PERP"], level=1, stop_signal=None)

    assert updates == [("BTC-PERP", Decimal("101"))]
    assert "BTC-PERP" in risk.last_mark_update
    assert event_store.metrics


def test_streaming_disabled_returns_none() -> None:
    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
    config.perps_enable_streaming = False
    registry = ServiceRegistry(
        config=config,
        broker=SimpleNamespace(),
        risk_manager=SimpleNamespace(),
        event_store=SimpleNamespace(append_metric=lambda *args, **kwargs: None),
    )
    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=registry.event_store,
        orders_store=None,
        broker=registry.broker,
        risk_manager=registry.risk_manager,
        symbols=("BTC-PERP",),
        bot_id="perps_bot",
        runtime_state=PerpsBotRuntimeState(["BTC-PERP"]),
        strategy_coordinator=None,
        set_running_flag=lambda _: None,
    )

    coordinator = TelemetryCoordinator(context)
    coordinator.update_context(context)

    assert coordinator._should_enable_streaming() is False
