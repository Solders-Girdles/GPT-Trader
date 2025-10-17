"""Utilities for deriving orchestration context data from a PerpsBot instance."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.service_registry import ServiceRegistry

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore


__all__ = ["BotStateExtractor"]


_UNSET = object()


class BotStateExtractor:
    """Provide validated, cached access to orchestration-related bot attributes."""

    __slots__ = (
        "_bot",
        "_config",
        "_runtime_state",
        "_product_cache",
        "_set_running_flag",
    )

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        self._config: Any | None = None
        self._runtime_state: Any = _UNSET
        self._product_cache: dict[str, Any] | None | object = _UNSET
        self._set_running_flag: Callable[[bool], None] | None | object = _UNSET

    def _safe_getattr(self, attr: str, default: Any = None) -> Any:
        try:
            return getattr(self._bot, attr)
        except (AttributeError, RuntimeError):
            return default

    @property
    def config(self) -> Any:
        """Return the bot configuration, defaulting to a PROD profile when missing."""

        if self._config is None:
            candidate = getattr(self._bot, "config", None)
            if isinstance(candidate, BotConfig):
                self._config = candidate
            elif candidate is not None:
                self._config = candidate
            else:
                self._config = BotConfig(profile=Profile.PROD)
        return self._config

    def service_registry(self, registry: ServiceRegistry | None = None) -> ServiceRegistry:
        """Resolve a registry instance aligned with the extracted configuration."""

        config = self.config
        bot_registry = getattr(self._bot, "registry", None)
        candidate: Any = registry if registry is not None else bot_registry

        if isinstance(candidate, ServiceRegistry):
            if candidate.config is config:
                return candidate
            return candidate.with_updates(config=config)

        extras: dict[str, Any] = {}
        runtime_settings = None
        if isinstance(bot_registry, ServiceRegistry):
            extras = dict(bot_registry.extras)
            runtime_settings = bot_registry.runtime_settings

        broker_candidate = _UNSET
        risk_candidate = _UNSET
        event_store_candidate = _UNSET
        orders_store_candidate = _UNSET

        if isinstance(bot_registry, ServiceRegistry):
            broker_candidate = getattr(bot_registry, "broker", _UNSET)
            risk_candidate = getattr(bot_registry, "risk_manager", _UNSET)
            event_store_candidate = getattr(bot_registry, "event_store", _UNSET)
            orders_store_candidate = getattr(bot_registry, "orders_store", _UNSET)
        elif hasattr(bot_registry, "__dict__") and isinstance(bot_registry.__dict__, dict):
            if "broker" in bot_registry.__dict__:
                broker_candidate = bot_registry.__dict__["broker"]
            if "risk_manager" in bot_registry.__dict__:
                risk_candidate = bot_registry.__dict__["risk_manager"]
            if "event_store" in bot_registry.__dict__:
                event_store_candidate = bot_registry.__dict__["event_store"]
            if "orders_store" in bot_registry.__dict__:
                orders_store_candidate = bot_registry.__dict__["orders_store"]

        if broker_candidate is _UNSET:
            broker_candidate = self.broker
        if risk_candidate is _UNSET:
            risk_candidate = self.risk_manager
        if event_store_candidate is _UNSET:
            event_store_candidate = self.event_store
        if orders_store_candidate is _UNSET:
            orders_store_candidate = self.orders_store

        if not extras and hasattr(bot_registry, "extras"):
            try:
                extras = dict(getattr(bot_registry, "extras"))
            except Exception:
                extras = {}

        return ServiceRegistry(
            config=config,
            event_store=event_store_candidate,
            orders_store=orders_store_candidate,
            broker=broker_candidate,
            risk_manager=risk_candidate,
            runtime_settings=runtime_settings,
            extras=extras,
        )

    def _normalize_symbols(self, source: Any) -> tuple[str, ...]:
        symbols = getattr(source, "symbols", None) or ()
        try:
            return tuple(symbols)
        except TypeError:
            return ()

    @property
    def event_store(self) -> EventStore | None:
        return cast("EventStore | None", self._safe_getattr("event_store"))

    @property
    def orders_store(self) -> OrdersStore | None:
        return cast("OrdersStore | None", self._safe_getattr("orders_store"))

    @property
    def broker(self) -> Any:
        return self._safe_getattr("broker")

    @property
    def risk_manager(self) -> Any:
        return self._safe_getattr("risk_manager")

    @property
    def runtime_state(self) -> Any:
        if self._runtime_state is _UNSET:
            self._runtime_state = getattr(self._bot, "runtime_state", None)
        return None if self._runtime_state is _UNSET else self._runtime_state

    @property
    def product_cache(self) -> dict[str, Any] | None:
        if self._product_cache is not _UNSET:
            return cast("dict[str, Any] | None", self._product_cache)

        cache = None
        state = getattr(self._bot, "_state", None)
        if state is not None:
            cache = getattr(state, "product_map", None)
        if cache is None:
            runtime_state = self.runtime_state
            if runtime_state is not None:
                cache = getattr(runtime_state, "product_map", None)

        self._product_cache = cache
        return cache

    @property
    def symbols(self) -> tuple[str, ...]:
        symbols = getattr(self.config, "symbols", None) or ()
        try:
            return tuple(symbols)
        except TypeError:
            return ()

    @property
    def bot_id(self) -> str:
        value = self._safe_getattr("bot_id", "perps_bot")
        if isinstance(value, str) and value:
            return value
        return "perps_bot"

    @property
    def config_controller(self) -> Any:
        return self._safe_getattr("config_controller")

    @property
    def strategy_orchestrator(self) -> Any:
        return self._safe_getattr("strategy_orchestrator")

    @property
    def strategy_coordinator(self) -> Any:
        return self._safe_getattr("strategy_coordinator")

    @property
    def execution_coordinator(self) -> Any:
        return self._safe_getattr("execution_coordinator")

    @property
    def session_guard(self) -> Any:
        return self._safe_getattr("_session_guard")

    @property
    def configuration_guardian(self) -> Any:
        return self._safe_getattr("configuration_guardian")

    @property
    def system_monitor(self) -> Any:
        return self._safe_getattr("system_monitor")

    @property
    def set_reduce_only_mode(self) -> Callable[[bool, str], None] | None:
        candidate = self._safe_getattr("set_reduce_only_mode")
        if callable(candidate):
            return candidate  # type: ignore[return-value]
        return None

    @property
    def shutdown_hook(self) -> Callable[[], Any] | None:
        candidate = self._safe_getattr("shutdown")
        if callable(candidate):
            return candidate  # type: ignore[return-value]
        return None

    @property
    def set_running_flag(self) -> Callable[[bool], None] | None:
        if self._set_running_flag is _UNSET:
            if hasattr(self._bot, "running"):

                def _set_running(value: bool) -> None:
                    setattr(self._bot, "running", value)

                self._set_running_flag = _set_running
            else:
                self._set_running_flag = None
        return None if self._set_running_flag is None else self._set_running_flag

    def coordinator_context_kwargs(
        self,
        registry: ServiceRegistry,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return keyword arguments for constructing a CoordinatorContext."""

        bot_config = getattr(self._bot, "config", None)
        symbol_source = bot_config if bot_config is not None else self.config
        bot_symbols = getattr(self._bot, "symbols", None)
        if bot_symbols is not None:
            try:
                symbols_value = tuple(bot_symbols)
            except TypeError:
                symbols_value = self._normalize_symbols(symbol_source)
        else:
            symbols_value = self._normalize_symbols(symbol_source)

        data: dict[str, Any] = {
            "config": bot_config if bot_config is not None else self.config,
            "registry": registry,
            "event_store": self.event_store,
            "orders_store": self.orders_store,
            "broker": registry.broker or self.broker,
            "risk_manager": registry.risk_manager or self.risk_manager,
            "symbols": symbols_value,
            "bot_id": self.bot_id,
            "runtime_state": self.runtime_state,
            "config_controller": self.config_controller,
            "strategy_orchestrator": self.strategy_orchestrator,
            "strategy_coordinator": self.strategy_coordinator,
            "execution_coordinator": self.execution_coordinator,
            "product_cache": self.product_cache,
            "session_guard": self.session_guard,
            "configuration_guardian": self.configuration_guardian,
            "system_monitor": self.system_monitor,
            "set_reduce_only_mode": self.set_reduce_only_mode,
            "shutdown_hook": self.shutdown_hook,
            "set_running_flag": self.set_running_flag,
        }

        if overrides:
            data.update(overrides)

        return data
