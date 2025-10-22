"""Shared fixtures for telemetry coordinator tests."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


@pytest.fixture
def make_context():
    """Factory fixture for creating CoordinatorContext instances."""

    def _make_context(
        *,
        broker: object | None = None,
        risk_manager: object | None = None,
        symbols: tuple[str, ...] = ("BTC-PERP",),
    ) -> CoordinatorContext:
        config = BotConfig(profile=Profile.PROD)
        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=Mock(),
            orders_store=Mock(),
        )
        runtime_state = PerpsBotRuntimeState(list(symbols))

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=registry.event_store,
            orders_store=registry.orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=symbols,
            bot_id=BOT_ID,
            runtime_state=runtime_state,
        )

    return _make_context
