"""Shared fixtures for system monitor positions tests."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.orchestration.system_monitor_positions import PositionReconciler

# Global iteration counter for controlled loop testing
_iteration_counter = {"count": 0}


def controlled_sleep(interval_seconds: int) -> None:
    """Controlled sleep that increments counter for deterministic loop testing."""
    _iteration_counter["count"] += 1


@pytest.fixture(autouse=True)
def patch_asyncio_functions():
    """Autouse patches to control async behavior in tests."""
    # Replace asyncio.to_thread with direct call for predictable behavior
    def sync_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "bot_v2.orchestration.system_monitor_positions.asyncio.to_thread",
        side_effect=sync_to_thread,
    ), patch(
        "bot_v2.orchestration.system_monitor_positions.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        yield


@pytest.fixture
def reset_iteration_counter():
    """Reset the iteration counter before each test."""
    _iteration_counter["count"] = 0
    yield
    _iteration_counter["count"] = 0


@pytest.fixture
def fake_event_store() -> MagicMock:
    """Mock event store for testing metric emission."""
    store = MagicMock()
    store.append_metric = MagicMock()
    return store


@pytest.fixture
def fake_plog() -> MagicMock:
    """Mock PLog for position change logging."""
    plog = MagicMock()
    plog.log_position_change = MagicMock()
    return plog


@pytest.fixture
def emit_metric_spy() -> MagicMock:
    """Spy for emit_metric function calls."""
    return MagicMock()


@pytest.fixture(autouse=True)
def patch_dependencies(fake_plog: MagicMock, emit_metric_spy: MagicMock):
    """Patch external dependencies for all tests."""
    with patch(
        "bot_v2.orchestration.system_monitor_positions._get_plog",
        return_value=fake_plog,
    ), patch(
        "bot_v2.orchestration.system_monitor_positions.emit_metric",
        emit_metric_spy,
    ):
        yield


@pytest.fixture
def fake_broker() -> MagicMock:
    """Mock broker with position listing capabilities."""
    broker = MagicMock()
    broker.list_positions = MagicMock(return_value=[])
    return broker


@pytest.fixture
def fake_runtime_state() -> SimpleNamespace:
    """Mock runtime state with last_positions tracking."""
    state = SimpleNamespace()
    state.last_positions = {}
    return state


@pytest.fixture
def fake_bot(fake_broker: MagicMock, fake_runtime_state: SimpleNamespace) -> SimpleNamespace:
    """Mock bot with running state and required dependencies."""
    bot = SimpleNamespace()
    bot.running = True
    bot.broker = fake_broker
    bot.runtime_state = fake_runtime_state

    # Helper to stop running after N iterations
    def stop_after_iterations(count: int):
        bot.running = _iteration_counter["count"] < count
        return bot.running

    bot.stop_after_iterations = stop_after_iterations
    return bot


@pytest.fixture
def sample_position() -> SimpleNamespace:
    """Sample position for testing."""
    pos = SimpleNamespace()
    pos.symbol = "BTC-PERP"
    pos.quantity = Decimal("0.5")
    pos.side = "long"
    return pos


@pytest.fixture
def sample_positions(sample_position: SimpleNamespace) -> list[SimpleNamespace]:
    """List of sample positions for testing."""
    second_pos = SimpleNamespace()
    second_pos.symbol = "ETH-PERP"
    second_pos.quantity = Decimal("1.0")
    second_pos.side = "short"

    return [sample_position, second_pos]


@pytest.fixture
def reconciler(fake_event_store: MagicMock) -> PositionReconciler:
    """PositionReconciler instance with mocked dependencies."""
    return PositionReconciler(event_store=fake_event_store, bot_id="test-bot")
