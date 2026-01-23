"""Tests for AccountTelemetryService broker_calls execution path."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


class _BrokerCallsSpy:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def __call__(self, func, *args, **kwargs):
        self.calls.append(func)
        return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_run_uses_broker_calls() -> None:
    broker = Mock()
    broker.get_key_permissions = Mock()
    broker.get_fee_schedule = Mock()
    broker.get_account_limits = Mock()
    broker.get_transaction_summary = Mock()
    broker.list_payment_methods = Mock()
    broker.list_portfolios = Mock()
    broker.get_server_time.return_value = None

    account_manager = Mock()
    account_manager.snapshot.return_value = {"balance": "1000"}

    broker_calls = _BrokerCallsSpy()
    service = AccountTelemetryService(
        broker=broker,
        account_manager=account_manager,
        event_store=Mock(),
        bot_id="test",
        profile="default",
        broker_calls=broker_calls,
    )

    published = asyncio.Event()

    def _publish_snapshot(snapshot: dict) -> None:
        published.set()

    service._publish_snapshot = Mock(side_effect=_publish_snapshot)

    task = asyncio.create_task(service.run(interval_seconds=10))
    await asyncio.wait_for(published.wait(), timeout=0.2)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    assert broker_calls.calls
