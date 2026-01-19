"""Tests for TradingEngine health check runner wiring."""

from __future__ import annotations

import asyncio

import pytest


def test_health_check_runner_initialized(engine):
    """Test that health check runner is initialized with engine."""
    from gpt_trader.monitoring.health_checks import HealthCheckRunner

    assert hasattr(engine, "_health_check_runner")
    assert isinstance(engine._health_check_runner, HealthCheckRunner)
    assert engine._health_check_runner._broker is engine.context.broker
    assert engine._health_check_runner._degradation_state is engine._degradation
    assert engine._health_check_runner._risk_manager is engine.context.risk_manager


@pytest.mark.asyncio
async def test_health_check_runner_started_and_stopped(engine, monkeypatch):
    """Test that health check runner starts/stops with engine lifecycle."""
    from unittest.mock import AsyncMock

    start_mock = AsyncMock()
    stop_mock = AsyncMock()

    monkeypatch.setattr(engine._health_check_runner, "start", start_mock)
    monkeypatch.setattr(engine._health_check_runner, "stop", stop_mock)

    engine._heartbeat.start = AsyncMock(return_value=None)
    engine._status_reporter.start = AsyncMock(return_value=None)
    engine._system_maintenance.start_prune_loop = AsyncMock(
        return_value=asyncio.create_task(asyncio.sleep(0))
    )
    engine._heartbeat.stop = AsyncMock()
    engine._status_reporter.stop = AsyncMock()
    engine._system_maintenance.stop = AsyncMock()

    engine.running = False

    async def mock_run_loop():
        pass

    monkeypatch.setattr(engine, "_run_loop", mock_run_loop)

    async def mock_monitor_ws_health():
        pass

    monkeypatch.setattr(engine, "_monitor_ws_health", mock_monitor_ws_health)

    monkeypatch.setattr(engine, "_should_enable_streaming", lambda: False)

    await engine.start_background_tasks()
    start_mock.assert_called_once()

    await engine.shutdown()
    stop_mock.assert_called_once()
