"""Helpers for constructing coordinator contexts from bot state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.service_registry import ServiceRegistry

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from bot_v2.orchestration.perps_bot import PerpsBot

__all__ = [
    "ensure_service_registry",
    "build_coordinator_context",
]


def _ensure_config(bot: PerpsBot) -> BotConfig:
    """Return a valid BotConfig for the provided bot."""

    config = getattr(bot, "config", None)
    if isinstance(config, BotConfig):
        return config
    return BotConfig(profile=Profile.PROD)


def _resolve_symbols(config: BotConfig) -> tuple[str, ...]:
    """Return the configured symbols as a tuple."""

    symbols = getattr(config, "symbols", None) or ()
    try:
        return tuple(symbols)
    except TypeError:
        return ()


def ensure_service_registry(
    bot: PerpsBot,
    registry: ServiceRegistry | None = None,
) -> ServiceRegistry:
    """Return a ServiceRegistry instance aligned with the bot configuration."""

    config = _ensure_config(bot)
    candidate = registry or getattr(bot, "registry", None)

    if isinstance(candidate, ServiceRegistry):
        if candidate.config is config:
            return candidate
        return candidate.with_updates(config=config)

    extras: dict[str, Any] = {}
    existing_registry = getattr(bot, "registry", None)
    if isinstance(existing_registry, ServiceRegistry):
        extras = dict(existing_registry.extras)

    return ServiceRegistry(
        config=config,
        event_store=getattr(bot, "event_store", None),
        orders_store=getattr(bot, "orders_store", None),
        broker=getattr(bot, "broker", None),
        risk_manager=getattr(bot, "risk_manager", None),
        runtime_settings=(
            getattr(existing_registry, "runtime_settings", None)
            if isinstance(existing_registry, ServiceRegistry)
            else None
        ),
        extras=extras,
    )


def build_coordinator_context(
    bot: PerpsBot,
    *,
    registry: ServiceRegistry | None = None,
    overrides: dict[str, Any] | None = None,
) -> CoordinatorContext:
    """Build a CoordinatorContext snapshot from a PerpsBot instance."""

    config = _ensure_config(bot)
    resolved_registry = ensure_service_registry(bot, registry)

    product_cache = None
    state = getattr(bot, "_state", None)
    if state is not None:
        product_cache = getattr(state, "product_map", None)
    if product_cache is None:
        runtime_state = getattr(bot, "runtime_state", None)
        if runtime_state is not None:
            product_cache = getattr(runtime_state, "product_map", None)

    context_data: dict[str, Any] = {
        "config": config,
        "registry": resolved_registry,
        "event_store": getattr(bot, "event_store", None),
        "orders_store": getattr(bot, "orders_store", None),
        "broker": resolved_registry.broker or getattr(bot, "broker", None),
        "risk_manager": resolved_registry.risk_manager or getattr(bot, "risk_manager", None),
        "symbols": _resolve_symbols(config),
        "bot_id": getattr(bot, "bot_id", "perps_bot"),
        "runtime_state": getattr(bot, "runtime_state", None),
        "config_controller": getattr(bot, "config_controller", None),
        "strategy_orchestrator": getattr(bot, "strategy_orchestrator", None),
        "strategy_coordinator": getattr(bot, "strategy_coordinator", None),
        "execution_coordinator": getattr(bot, "execution_coordinator", None),
        "product_cache": product_cache,
        "session_guard": getattr(bot, "_session_guard", None),
        "configuration_guardian": getattr(bot, "configuration_guardian", None),
        "system_monitor": getattr(bot, "system_monitor", None),
        "set_reduce_only_mode": getattr(bot, "set_reduce_only_mode", None),
        "shutdown_hook": getattr(bot, "shutdown", None),
    }

    if hasattr(bot, "running"):

        def _set_running(value: bool) -> None:
            setattr(bot, "running", value)

        context_data["set_running_flag"] = _set_running  # type: ignore[assignment]
    else:
        context_data["set_running_flag"] = None

    if overrides:
        context_data.update(overrides)

    return CoordinatorContext(**context_data)
