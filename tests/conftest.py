"""
Pytest configuration to support async test functions without requiring
pytest-asyncio. We map asyncio-marked tests to anyio and provide a
fallback executor for plain async defs.
"""

import asyncio
import inspect
import time

import pytest

pytest_plugins = [
    "tests.fixtures.advanced_execution",
]

from tests.fixtures.behavioral import FakeClock
from tests.fixtures.environment import (  # noqa: F401
    env_override,
    hvac_stub,
    patched_runtime_settings,
    runtime_settings_factory,
    temp_home,
    yahoo_provider_stub,
)
from tests.fixtures.monitoring import (  # noqa: F401
    advance_time,
    alert_manager,
    alert_recorder,
    frozen_time,
    monitoring_collectors,
)
from tests.fixtures.optimization import (  # noqa: F401
    backtest_metrics_factory,
    fake_backtest_runner,
    ohlc_data_factory,
    optimization_workspace,
    seeded_ohlc_sets,
)


def pytest_collection_modifyitems(items):
    for item in items:
        # If tests use pytest-asyncio marker but the plugin isn't installed,
        # forward the marker to anyio which is available in this environment.
        if item.get_closest_marker("asyncio") is not None:
            item.add_marker(pytest.mark.anyio)
        # Centralized skip rules removed in favor of explicit per-file markers.


def pytest_pyfunc_call(pyfuncitem):
    """Fallback runner for async test functions.

    If a test function is a coroutine and no plugin has handled it, run
    it using asyncio.run so tests don't error on missing plugins.
    """
    testfunc = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunc):
        # Filter kwargs to only those accepted by the function signature
        sig = inspect.signature(testfunc)
        accepted = set(sig.parameters.keys())
        kwargs = {k: v for k, v in pyfuncitem.funcargs.items() if k in accepted}
        coro = testfunc(**kwargs)
        asyncio.run(coro)
        return True
    return None


@pytest.fixture
def fake_clock(monkeypatch):
    """Provide a deterministic clock and patch time/asyncio helpers."""
    clock = FakeClock()
    real_async_sleep = asyncio.sleep
    real_sleep = time.sleep

    # Ensure async sleep yields control at least once per call.
    clock.set_async_yield(lambda: real_async_sleep(0))
    clock.set_thread_yield(lambda: real_sleep(0))

    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    monkeypatch.setattr(asyncio, "sleep", clock.async_sleep)

    yield clock

    # Restore asyncio.sleep implicitly via monkeypatch teardown.
