import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_v2.orchestration.lifecycle_manager import LifecycleManager


class BackgroundProbe:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, *args: Any, **kwargs: Any) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def make_bot(*, dry_run: bool, telemetry_enabled: bool) -> SimpleNamespace:
    probes = {
        "guards": BackgroundProbe(),
        "orders": BackgroundProbe(),
        "positions": BackgroundProbe(),
        "account": BackgroundProbe(),
    }

    runtime_coordinator = SimpleNamespace(
        reconcile_state_on_startup=AsyncMock(),
    )

    telemetry_coordinator = SimpleNamespace(run_account_telemetry=probes["account"].run)

    execution_coordinator = SimpleNamespace()

    system_monitor = SimpleNamespace(
        run_position_reconciliation=probes["positions"].run,
        write_health_status=MagicMock(),
        check_config_updates=MagicMock(),
    )

    account_telemetry = SimpleNamespace(
        supports_snapshots=lambda: telemetry_enabled,
    )

    bot = SimpleNamespace(
        config=SimpleNamespace(
            profile=SimpleNamespace(value="TEST"),
            dry_run=dry_run,
            account_telemetry_interval=5,
            update_interval=0.01,
        ),
        runtime_coordinator=runtime_coordinator,
        telemetry_coordinator=telemetry_coordinator,
        execution_coordinator=execution_coordinator,
        system_monitor=system_monitor,
        account_telemetry=account_telemetry,
        running=False,
        probes=probes,
    )

    class RegistryStub:
        def __init__(self) -> None:
            self.start_calls = 0
            self.shutdown_calls = 0
            self.tasks: list[asyncio.Task[Any]] = []

        def initialize_all(self) -> SimpleNamespace:
            return SimpleNamespace(
                registry=None,
                broker=None,
                risk_manager=None,
            )

        async def start_all_background_tasks(self) -> list[asyncio.Task[Any]]:
            self.start_calls += 1
            tasks = [
                asyncio.create_task(probes["guards"].run()),
                asyncio.create_task(probes["orders"].run()),
                asyncio.create_task(probes["account"].run()),
            ]
            self.tasks.extend(tasks)
            return tasks

        async def shutdown_all(self) -> None:
            self.shutdown_calls += 1

    bot._coordinator_registry = RegistryStub()
    bot._coordinator_context = SimpleNamespace()

    bot.run_cycle_calls = 0

    async def run_cycle() -> None:
        bot.run_cycle_calls += 1
        # Allow other tasks to start before exiting the loop
        await asyncio.sleep(0)
        bot.running = False

    bot.run_cycle = run_cycle

    return bot


@pytest.mark.asyncio
async def test_run_single_cycle_dry_run_skips_background_tasks() -> None:
    bot = make_bot(dry_run=True, telemetry_enabled=True)
    manager = LifecycleManager(bot)

    await manager.run(single_cycle=True)

    assert bot.run_cycle_calls == 1
    assert bot.runtime_coordinator.reconcile_state_on_startup.await_count == 0
    assert bot._coordinator_registry.start_calls == 0
    assert not bot.probes["guards"].started.is_set()
    assert not bot.probes["orders"].started.is_set()
    assert not bot.probes["positions"].started.is_set()
    assert not bot.probes["account"].started.is_set()
    assert bot._coordinator_registry.shutdown_calls == 1


@pytest.mark.asyncio
async def test_run_starts_and_cancels_background_tasks() -> None:
    bot = make_bot(dry_run=False, telemetry_enabled=True)
    manager = LifecycleManager(bot)

    await manager.run(single_cycle=False)

    bot.runtime_coordinator.reconcile_state_on_startup.assert_awaited()
    assert bot._coordinator_registry.start_calls == 1
    for name in ("guards", "orders", "positions", "account"):
        assert bot.probes[name].started.is_set()
        assert bot.probes[name].cancelled.is_set()
    assert bot.system_monitor.write_health_status.call_count >= 1
    assert bot.system_monitor.check_config_updates.call_count >= 1
    assert bot._coordinator_registry.shutdown_calls == 1
