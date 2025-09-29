"""Tests for alert channel fallbacks."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from bot_v2.monitoring.alerts import Alert, AlertChannel, AlertSeverity


@pytest.mark.asyncio
async def test_base_alert_channel_logs_and_returns_false(caplog):
    channel = AlertChannel()
    alert = Alert(
        timestamp=datetime.utcnow(),
        source="unit-test",
        severity=AlertSeverity.INFO,
        title="Generic Alert",
        message="Just a drill",
    )

    with caplog.at_level(logging.WARNING):
        result = await channel.send(alert)

    assert result is False
    assert any("no concrete send implementation" in record.message for record in caplog.records)
