from __future__ import annotations

from types import SimpleNamespace

from bot_v2.orchestration.bot_state_extraction import BotStateExtractor
from bot_v2.orchestration.configuration import BotConfig, Profile
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

    async def shutdown(self) -> None:  # pragma: no cover - behaviour validated indirectly
        self.shutdown_called = True


class MinimalBot:
    def __init__(self) -> None:
        self.config = BotConfig(profile=Profile.DEV, symbols=["ETH-USD"])
        self.broker = "broker-min"
        self.risk_manager = "risk-min"
        self.event_store = "event"
        self.orders_store = "orders"


def test_service_registry_reuses_existing_instance() -> None:
    bot = StubBot()
    extractor = BotStateExtractor(bot)

    registry = extractor.service_registry()

    assert registry is bot.registry
    assert registry.config is bot.config


def test_service_registry_constructs_default_when_missing() -> None:
    bot = MinimalBot()
    extractor = BotStateExtractor(bot)

    registry = extractor.service_registry()

    assert isinstance(registry, ServiceRegistry)
    assert registry.config is bot.config
    assert registry.broker == bot.broker
    assert registry.risk_manager == bot.risk_manager
    assert registry.event_store == bot.event_store
    assert registry.orders_store == bot.orders_store


def test_coordinator_context_kwargs_exposes_expected_values() -> None:
    bot = StubBot()
    extractor = BotStateExtractor(bot)
    registry = extractor.service_registry()
    override_exec = object()

    data = extractor.coordinator_context_kwargs(
        registry,
        overrides={"execution_coordinator": override_exec},
    )

    assert data["config"] is bot.config
    assert data["registry"] is registry
    assert data["product_cache"] == bot._state.product_map
    assert data["symbols"] == ("BTC-USD",)
    assert data["execution_coordinator"] is override_exec
    assert callable(data["set_reduce_only_mode"])
    assert callable(data["set_running_flag"])

    setter = data["set_running_flag"]
    assert setter is not None
    setter(False)
    assert bot.running is False
