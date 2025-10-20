"""
Fixtures for lifecycle_manager tests.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def fake_coordinator_registry():
    """Create fake coordinator registry with async methods."""
    registry = Mock()
    registry.initialize_all = Mock()
    registry.start_all_background_tasks = AsyncMock(return_value=[])
    registry.shutdown_all = AsyncMock()
    return registry


@pytest.fixture
def fake_strategy_orchestrator():
    """Create fake strategy orchestrator."""
    orchestrator = Mock()
    orchestrator.init_strategy = Mock()
    return orchestrator


@pytest.fixture
def fake_system_monitor():
    """Create fake system monitor."""
    monitor = Mock()
    monitor.run_position_reconciliation = AsyncMock()
    monitor.write_health_status = Mock()
    monitor.check_config_updates = Mock()
    return monitor


@pytest.fixture
def fake_runtime_coordinator():
    """Create fake runtime coordinator."""
    coordinator = Mock()
    coordinator.reconcile_state_on_startup = AsyncMock()
    return coordinator


@pytest.fixture
def fake_bot(
    fake_coordinator_registry,
    fake_strategy_orchestrator,
    fake_system_monitor,
    fake_runtime_coordinator,
):
    """Create fake PerpsBot instance."""
    bot = Mock()
    bot.config = SimpleNamespace(
        profile=SimpleNamespace(value="TEST"),
        dry_run=False,
        update_interval=0.01,
    )
    bot.symbols = ["BTC-PERP"]
    bot.running = False
    bot.run_cycle = AsyncMock()
    bot._coordinator_registry = fake_coordinator_registry
    bot._coordinator_context = SimpleNamespace()
    bot.strategy_orchestrator = fake_strategy_orchestrator
    bot.system_monitor = fake_system_monitor
    bot.runtime_coordinator = fake_runtime_coordinator
    bot.registry = SimpleNamespace(extras={})
    return bot
