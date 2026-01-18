"""Unit tests for status reporter file output and integration behavior."""

from __future__ import annotations

import asyncio
import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter


class TestStatusReporterFileOutput:
    """Tests for StatusReporter file output."""

    @pytest.mark.asyncio
    async def test_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="test-bot",
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000.123"))
            reporter.record_cycle()

            await asyncio.sleep(0.15)

            try:
                with open(status_file) as f:
                    data = json.load(f)

                assert data["bot_id"] == "test-bot"
                assert data["engine"]["cycle_count"] == 1
                assert "BTC-USD" in data["market"]["last_prices"]
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_handles_decimal_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000.12345678"))
            reporter.update_positions({"BTC-PERP": {"unrealized_pnl": Decimal("123.456")}})

            await asyncio.sleep(0.15)

            try:
                with open(status_file) as f:
                    data = json.load(f)

                assert data["market"]["last_prices"]["BTC-USD"] == "50000.12345678"
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_atomic_write(self) -> None:
        """Test that writes are atomic (no partial files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                observer_interval=0.02,
                file_write_interval=0.02,
            )

            await reporter.start()

            try:
                for _ in range(10):
                    reporter.record_cycle()
                    await asyncio.sleep(0.03)

                with open(status_file) as f:
                    data = json.load(f)
                assert "timestamp" in data
            finally:
                await reporter.stop()


class TestStatusReporterIntegration:
    """Integration tests for StatusReporter."""

    @pytest.mark.asyncio
    async def test_full_status_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="integration-test",
                observer_interval=0.05,
                file_write_interval=0.05,
            )

            await reporter.start()

            reporter.update_price("BTC-USD", Decimal("50000"))
            reporter.update_price("ETH-USD", Decimal("3000"))
            reporter.record_cycle()
            reporter.update_positions(
                {"BTC-PERP": {"quantity": Decimal("1"), "unrealized_pnl": Decimal("50")}}
            )

            await asyncio.sleep(0.15)

            with open(status_file) as f:
                data = json.load(f)

            assert data["bot_id"] == "integration-test"
            assert data["engine"]["running"] is True
            assert data["engine"]["cycle_count"] == 1
            assert len(data["market"]["symbols"]) == 2
            assert data["positions"]["count"] == 1

            await reporter.stop()

            with open(status_file) as f:
                final_data = json.load(f)

            assert final_data["engine"]["running"] is False
