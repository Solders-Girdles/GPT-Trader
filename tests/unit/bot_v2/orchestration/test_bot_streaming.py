from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.telemetry_coordinator import TelemetryCoordinator


def test_stream_loop_updates_mark_window_unit() -> None:
    updates: list[tuple[str, Decimal]] = []

    class StubStrategy:
        def update_mark_window(self, symbol: str, mark: Decimal) -> None:
            updates.append((symbol, mark))

    class StubRisk:
        def __init__(self) -> None:
            self.last_mark_update: dict[str, object] = {}

    class StubEventStore:
        def __init__(self) -> None:
            self.metrics: list[tuple[str, dict[str, object]]] = []

        def append_metric(self, *args, **kwargs) -> None:
            if kwargs:
                bot_id = kwargs.get("bot_id")
                metrics = kwargs.get("metrics")
            else:
                bot_id, metrics = args
            assert bot_id is not None and metrics is not None
            self.metrics.append((bot_id, dict(metrics)))

    class StubBroker:
        def stream_orderbook(self, symbols: list[str], *, level: int) -> list[dict[str, object]]:
            return [
                {
                    "product_id": symbols[0],
                    "best_bid": "100",
                    "best_ask": "102",
                }
            ]

    config = BotConfig(profile=Profile.PROD)
    config.perps_enable_streaming = True
    config.perps_stream_level = 1
    config.short_ma = 2
    config.long_ma = 3
    config.symbols = ["BTC-PERP"]

    registry = ServiceRegistry(
        config=config,
        broker=StubBroker(),
        risk_manager=StubRisk(),
        event_store=StubEventStore(),
    )

    bot = SimpleNamespace(
        bot_id="perps_bot",
        config=config,
        registry=registry,
        symbols=["BTC-PERP"],
        broker=registry.broker,
        strategy_coordinator=StubStrategy(),
        event_store=registry.event_store,
        risk_manager=registry.risk_manager,
    )

    coordinator = TelemetryCoordinator(bot)
    coordinator._market_monitor = None  # type: ignore[attr-defined]

    coordinator._run_stream_loop(bot.symbols, level=1, stop_signal=None)

    assert updates == [("BTC-PERP", Decimal("101"))]
    assert "BTC-PERP" in bot.risk_manager.last_mark_update
    assert bot.event_store.metrics


@pytest.mark.asyncio
async def test_background_stream_task_emits_updates() -> None:
    updates: list[tuple[str, Decimal]] = []

    class StubStrategy:
        def update_mark_window(self, symbol: str, mark: Decimal) -> None:
            updates.append((symbol, mark))

    class StubRisk:
        def __init__(self) -> None:
            self.last_mark_update: dict[str, object] = {}

    class StubEventStore:
        def __init__(self) -> None:
            self.metrics: list[tuple[str, dict[str, object]]] = []

        def append_metric(self, *args, **kwargs) -> None:
            if kwargs:
                bot_id = kwargs.get("bot_id")
                metrics = kwargs.get("metrics")
            else:
                bot_id, metrics = args
            self.metrics.append((bot_id, dict(metrics)))

    class StubBroker:
        def __init__(self) -> None:
            self.orderbook_calls: list[tuple[tuple[str, ...], int]] = []

        def stream_orderbook(self, symbols: list[str], *, level: int):
            self.orderbook_calls.append((tuple(symbols), level))
            yield {"product_id": symbols[0], "best_bid": "100", "best_ask": "101"}

    config = BotConfig(profile=Profile.PROD)
    config.perps_enable_streaming = True
    config.perps_stream_level = 1
    config.short_ma = 2
    config.long_ma = 3
    config.symbols = ["BTC-PERP"]

    event_store = StubEventStore()
    risk_manager = StubRisk()
    broker = StubBroker()

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
    )

    bot = SimpleNamespace(
        bot_id="perps_bot",
        config=config,
        registry=registry,
        symbols=["BTC-PERP"],
        broker=broker,
        strategy_coordinator=StubStrategy(),
        event_store=event_store,
        risk_manager=risk_manager,
    )

    coordinator = TelemetryCoordinator(bot)
    coordinator._market_monitor = None  # type: ignore[attr-defined]

    coordinator.bootstrap()
    tasks = await coordinator.start_background_tasks()
    assert tasks
    await asyncio.gather(*tasks)

    assert updates == [("BTC-PERP", Decimal("100.5"))]
    assert "BTC-PERP" in bot.risk_manager.last_mark_update
    assert bot.event_store.metrics
    assert bot.broker.orderbook_calls == [(("BTC-PERP",), 1)]

    await coordinator.shutdown()
