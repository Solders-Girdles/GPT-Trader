"""Tests for TradingEngine runtime guard sweep loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_runtime_guard_sweep_calls_guard_manager(engine, monkeypatch):
    """Test that runtime guard sweep calls GuardManager.run_runtime_guards."""
    engine._guard_manager = MagicMock()
    engine.running = True

    async def _sleep(_):
        engine.running = False
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    engine._guard_manager.run_runtime_guards.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_guard_sweep_skips_without_guard_manager(engine, monkeypatch):
    """Test that runtime guard sweep handles missing guard manager gracefully."""
    engine._guard_manager = None
    engine.running = True

    async def _sleep(_):
        engine.running = False
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()


@pytest.mark.asyncio
async def test_runtime_guard_sweep_uses_config_interval(engine, monkeypatch):
    """Test that runtime guard sweep uses configured interval."""
    engine._guard_manager = MagicMock()
    engine.running = True
    engine.context.config.runtime_guard_interval = 30

    sleep_intervals = []

    async def _sleep(interval):
        sleep_intervals.append(interval)
        engine.running = False
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    assert sleep_intervals == [30]


@pytest.mark.asyncio
async def test_runtime_guard_sweep_handles_exceptions(engine, monkeypatch):
    """Test that runtime guard sweep continues after generic exceptions."""
    engine._guard_manager = MagicMock()
    engine._guard_manager.run_runtime_guards.side_effect = RuntimeError("test error")
    engine.running = True

    call_count = 0

    async def _sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            engine.running = False
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    assert engine._guard_manager.run_runtime_guards.call_count == 2
