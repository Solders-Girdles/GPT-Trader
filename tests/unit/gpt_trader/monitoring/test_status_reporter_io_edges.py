"""Edge-case tests for StatusReporter serialization and IO."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_trader.monitoring.status_reporter import DecimalEncoder, StatusReporter


@pytest.mark.asyncio
async def test_write_status_creates_directories_and_writes_json(tmp_path: Path) -> None:
    status_file = tmp_path / "missing" / "dir" / "status.json"
    reporter = StatusReporter(status_file=str(status_file))

    await reporter._write_status_to_file()

    assert status_file.exists()
    data = json.loads(status_file.read_text())
    assert "engine" in data
    assert "market" in data


def test_decimal_encoder_handles_nested_decimals() -> None:
    reporter = StatusReporter()
    reporter.update_positions(
        {
            "BTC-PERP": {
                "quantity": Decimal("1.5"),
                "unrealized_pnl": Decimal("0.1"),
                "meta": {"nested": Decimal("2.5")},
            }
        }
    )
    reporter._update_status()
    payload = reporter._serialize_status()

    encoded = json.dumps(payload, cls=DecimalEncoder)
    decoded = json.loads(encoded)
    assert decoded["positions"]["positions"]["BTC-PERP"]["meta"]["nested"] == "2.5"


@pytest.mark.asyncio
async def test_write_status_handles_write_errors(tmp_path: Path) -> None:
    status_file = tmp_path / "status.json"
    reporter = StatusReporter(status_file=str(status_file))

    with patch("gpt_trader.monitoring.status_reporter.json.dump", side_effect=OSError("boom")):
        await reporter._write_status_to_file()

    assert not status_file.exists()


def test_serialize_status_includes_expected_keys() -> None:
    reporter = StatusReporter()
    payload = reporter._serialize_status()

    for key in (
        "bot_id",
        "timestamp",
        "engine",
        "market",
        "positions",
        "orders",
        "trades",
        "account",
        "strategy",
        "risk",
        "system",
        "heartbeat",
        "websocket",
    ):
        assert key in payload
