from __future__ import annotations

from types import SimpleNamespace

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.context_builder import (
    build_coordinator_context,
    ensure_service_registry,
)
from bot_v2.orchestration.service_registry import ServiceRegistry


class StubBot:
    def __init__(self) -> None:
        self.config = BotConfig(profile=Profile.PROD, symbols=["BTC-USD"])
        self.registry = ServiceRegistry(
            config=self.config,
            broker="broker",
            risk_manager="risk",
            event_store="event-store",
            orders_store="orders-store",
            extras={"account_manager": "am"},
        )
        self.event_store = "event-store"
        self.orders_store = "orders-store"
        self.broker = "broker"
        self.risk_manager = "risk"
        self.runtime_state = SimpleNamespace(product_map={"BTC-USD": "runtime-cache"})
        self._state = SimpleNamespace(product_map={"BTC-USD": "state-cache"})
        self._session_guard = "guard"
        self.configuration_guardian = "guardian"
        self.system_monitor = "monitor"
        self.strategy_orchestrator = "orchestrator"
        self.execution_coordinator = "exec-coordinator"
        self.strategy_coordinator = "strategy-coordinator"
        self.bot_id = "perps_bot"
        self.running = True
        self._reduce_only_events: list[tuple[bool, str]] = []
        self.shutdown_called = False

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        self._reduce_only_events.append((enabled, reason))

    async def shutdown(self) -> None:  # pragma: no cover - exercised via identity checks
        self.shutdown_called = True


class MinimalBot:
    def __init__(self) -> None:
        self.config = BotConfig(profile=Profile.DEV, symbols=["ETH-USD"])
        self.broker = "broker-min"
        self.risk_manager = "risk-min"
        self.event_store = "event"
        self.orders_store = "orders"


def test_build_coordinator_context_reuses_existing_registry() -> None:
    bot = StubBot()
    ctx = build_coordinator_context(bot)

    assert ctx.registry is bot.registry
    assert ctx.symbols == ("BTC-USD",)
    assert ctx.product_cache == bot._state.product_map
    assert ctx.broker == "broker"
    assert ctx.risk_manager == "risk"
    assert ctx.session_guard == bot._session_guard
    assert ctx.configuration_guardian == bot.configuration_guardian
    assert ctx.system_monitor == bot.system_monitor
    assert callable(ctx.set_reduce_only_mode)
    assert ctx.set_reduce_only_mode.__self__ is bot  # type: ignore[attr-defined]
    ctx.set_reduce_only_mode(True, "test")  # type: ignore[call-arg]
    assert bot._reduce_only_events[-1] == (True, "test")

    assert ctx.shutdown_hook.__self__ is bot  # type: ignore[attr-defined]
    assert callable(ctx.set_running_flag)

    ctx.set_running_flag(False)
    assert bot.running is False


def test_ensure_service_registry_creates_default_from_bot_attributes() -> None:
    bot = MinimalBot()
    registry = ensure_service_registry(bot, None)

    assert isinstance(registry, ServiceRegistry)
    assert registry.config is bot.config
    assert registry.broker == bot.broker
    assert registry.risk_manager == bot.risk_manager
    assert registry.event_store == bot.event_store
    assert registry.orders_store == bot.orders_store


def test_build_coordinator_context_allows_overrides() -> None:
    bot = StubBot()
    override_exec = object()

    ctx = build_coordinator_context(bot, overrides={"execution_coordinator": override_exec})

    assert ctx.execution_coordinator is override_exec
    # ensure other context data remains intact
    assert ctx.registry is bot.registry
    assert ctx.product_cache == bot._state.product_map


def test_build_coordinator_context_handles_missing_running_attribute() -> None:
    bot = MinimalBot()
    delattr(bot, "orders_store")
    # intentionally omit running attribute
    ctx = build_coordinator_context(bot)

    assert ctx.set_running_flag is None
    assert ctx.registry.config is bot.config
