"""Compatibility shim exposing the legacy ``app.container`` module path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import bot_v2.app.container as _impl
from bot_v2.orchestration.runtime_settings import load_runtime_settings  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    pass

PerpsBot: Any | None = getattr(_impl, "PerpsBot", None)
ConfigurationGuardian: Any | None = getattr(_impl, "ConfigurationGuardian", None)

ApplicationContainer = _impl.ApplicationContainer
create_application_container = _impl.create_application_container
MarketDataService = _impl.MarketDataService
ProductCatalog = _impl.ProductCatalog
create_brokerage = _impl.create_brokerage

# Ensure legacy patch targets continue to work by aliasing attributes on the upstream module.
_impl.MarketDataService = MarketDataService
_impl.ProductCatalog = ProductCatalog
_impl.create_brokerage = create_brokerage
_impl.load_runtime_settings = load_runtime_settings

__all__ = [
    "ApplicationContainer",
    "create_application_container",
    "MarketDataService",
    "ProductCatalog",
    "create_brokerage",
    "PerpsBot",
    "ConfigurationGuardian",
    "load_runtime_settings",
]


def __getattr__(name: str):
    if name == "PerpsBot":
        from bot_v2.orchestration.perps_bot import PerpsBot as _PerpsBot

        globals()["PerpsBot"] = _PerpsBot
        setattr(_impl, "PerpsBot", _PerpsBot)
        return _PerpsBot
    if name == "ConfigurationGuardian":
        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        globals()["ConfigurationGuardian"] = _ConfigurationGuardian
        setattr(_impl, "ConfigurationGuardian", _ConfigurationGuardian)
        return _ConfigurationGuardian
    raise AttributeError(name)
