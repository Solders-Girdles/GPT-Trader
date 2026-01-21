"""Tests for AccountTelemetryService async run loop."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


class TestRunAsync:
    """Tests for run async method."""

    @pytest.mark.asyncio
    async def test_run_exits_when_snapshots_not_supported(self) -> None:
        """Test run exits immediately when broker doesn't support snapshots."""
        broker = Mock(spec=[])  # No required methods

        service = AccountTelemetryService(
            broker=broker,
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        result = await asyncio.wait_for(service.run(interval_seconds=1), timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_run_collects_and_publishes(self) -> None:
        """Test run collects snapshot and publishes."""
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

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
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

        service._publish_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_run_handles_collection_error(self) -> None:
        """Test run handles errors during collection gracefully."""
        broker = Mock()
        broker.get_key_permissions = Mock()
        broker.get_fee_schedule = Mock()
        broker.get_account_limits = Mock()
        broker.get_transaction_summary = Mock()
        broker.list_payment_methods = Mock()
        broker.list_portfolios = Mock()

        account_manager = Mock()
        account_manager.snapshot.side_effect = RuntimeError("Collection error")

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
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

        account_manager.snapshot.assert_called()
        service._publish_snapshot.assert_called()
