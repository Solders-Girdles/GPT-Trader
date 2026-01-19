"""Chaos tests for TradingEngine broker-outage degradation."""

from __future__ import annotations

import pytest
from tests.support.chaos import ChaosBroker, broker_read_failures_scenario

pytest_plugins = ["strategy_engine_chaos_fixtures"]


class TestBrokerOutageDegradation:
    @pytest.mark.asyncio
    async def test_broker_failures_trigger_pause_after_threshold(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        engine.context.broker = ChaosBroker(mock_broker, broker_read_failures_scenario(times=3))
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 1
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 2
        await engine._fetch_total_equity({})
        assert engine._degradation.is_paused() and "broker_outage" in (
            engine._degradation.get_pause_reason() or ""
        )

    @pytest.mark.asyncio
    async def test_successful_broker_call_resets_counter(self, engine, mock_broker) -> None:
        engine._degradation._broker_failures = 2
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 0
