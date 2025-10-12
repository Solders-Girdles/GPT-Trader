from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from bot_v2.orchestration.configuration import Profile
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

    bot = SimpleNamespace(
        bot_id="perps_bot",
        symbols=["BTC-PERP"],
        broker=StubBroker(),
        strategy_coordinator=StubStrategy(),
        event_store=StubEventStore(),
        risk_manager=StubRisk(),
    )

    coordinator = TelemetryCoordinator(bot)
    coordinator._market_monitor = None  # type: ignore[attr-defined]

    coordinator._run_stream_loop(bot.symbols, level=1)

    assert updates == [("BTC-PERP", Decimal("101"))]
    assert "BTC-PERP" in bot.risk_manager.last_mark_update
    assert bot.event_store.metrics


def test_background_stream_thread_emits_updates(monkeypatch) -> None:
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

    bot = SimpleNamespace(
        bot_id="perps_bot",
        symbols=["BTC-PERP"],
        config=SimpleNamespace(
            perps_enable_streaming=True, perps_stream_level=1, profile=Profile.PROD
        ),
        broker=StubBroker(),
        strategy_coordinator=StubStrategy(),
        event_store=StubEventStore(),
        risk_manager=StubRisk(),
    )

    coordinator = TelemetryCoordinator(bot)
    coordinator._market_monitor = None  # type: ignore[attr-defined]

    coordinator.start_streaming_background()
    assert coordinator._ws_thread is not None
    coordinator._ws_thread.join(timeout=1.0)
    assert not coordinator._ws_thread.is_alive()

    assert updates == [("BTC-PERP", Decimal("100.5"))]
    assert "BTC-PERP" in bot.risk_manager.last_mark_update
    assert bot.event_store.metrics
    assert bot.broker.orderbook_calls == [(("BTC-PERP",), 1)]

    coordinator.stop_streaming_background()
