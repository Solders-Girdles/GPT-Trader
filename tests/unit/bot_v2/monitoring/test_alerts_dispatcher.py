from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime
from typing import Any

import pytest

from bot_v2.monitoring import alerts


@pytest.fixture
def sample_alert() -> alerts.Alert:
    return alerts.Alert(
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        source="system",
        severity=alerts.AlertSeverity.ERROR,
        title="Disk Full",
        message="Root partition at 95%",
        context={"disk": "/", "percent": 95},
        alert_id="abcd1234",
    )


def test_alert_to_dict(sample_alert: alerts.Alert) -> None:
    payload = sample_alert.to_dict()
    assert payload["alert_id"] == "abcd1234"
    assert payload["severity"] == "error"
    assert payload["context"]["disk"] == "/"


@pytest.mark.asyncio
async def test_alert_channel_send_threshold(sample_alert: alerts.Alert) -> None:
    channel = alerts.AlertChannel(min_severity=alerts.AlertSeverity.CRITICAL)
    # Should skip because alert severity below threshold
    sent = await channel.send(sample_alert)
    assert not sent

    # Trigger error handling path by forcing _send_impl to raise
    class BrokenChannel(alerts.AlertChannel):
        async def _send_impl(self, alert: alerts.Alert) -> bool:  # type: ignore[override]
            raise RuntimeError("boom")

    broken = BrokenChannel(min_severity=alerts.AlertSeverity.ERROR)
    assert not await broken.send(sample_alert)


class RecordingChannel(alerts.AlertChannel):
    def __init__(
        self,
        name: str,
        *,
        should_fail: bool = False,
        min_severity: alerts.AlertSeverity = alerts.AlertSeverity.INFO,
    ) -> None:
        super().__init__(min_severity=min_severity)
        self.name = name
        self.should_fail = should_fail
        self.received: list[alerts.Alert] = []

    async def _send_impl(self, alert: alerts.Alert) -> bool:  # type: ignore[override]
        if self.should_fail:
            raise RuntimeError("network error")
        self.received.append(alert)
        return True


@pytest.mark.asyncio
async def test_alert_dispatcher_dispatch_flow(sample_alert: alerts.Alert) -> None:
    dispatcher = alerts.AlertDispatcher()
    dispatcher.add_channel(
        "primary", RecordingChannel("primary", min_severity=alerts.AlertSeverity.ERROR)
    )
    dispatcher.add_channel("secondary", RecordingChannel("secondary", should_fail=True))
    dispatcher.add_channel(
        "debug", RecordingChannel("debug", min_severity=alerts.AlertSeverity.CRITICAL)
    )

    results = await dispatcher.dispatch(sample_alert)

    # Only primary should succeed, secondary should fail, debug skipped (threshold)
    assert results["primary"] is True
    assert results["secondary"] is False
    assert len(dispatcher.alert_history) == 1

    recent = dispatcher.get_recent_alerts(severity=alerts.AlertSeverity.ERROR)
    assert recent and recent[0].title == "Disk Full"


def test_dispatcher_from_config_builds_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    added: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class DummyChannel(alerts.AlertChannel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            added.append((self.__class__.__name__.lower(), args, kwargs))

        async def _send_impl(self, alert: alerts.Alert) -> bool:  # type: ignore[override]
            return True

    monkeypatch.setattr(alerts, "SlackChannel", type("SlackStub", (DummyChannel,), {}))
    monkeypatch.setattr(alerts, "PagerDutyChannel", type("PagerDutyStub", (DummyChannel,), {}))
    monkeypatch.setattr(alerts, "EmailChannel", type("EmailStub", (DummyChannel,), {}))
    monkeypatch.setattr(alerts, "WebhookChannel", type("WebhookStub", (DummyChannel,), {}))

    config = {
        "slack_webhook_url": "https://hooks.slack.test/123",
        "slack_min_severity": "warning",
        "pagerduty_api_key": "pd-key",
        "pagerduty_min_severity": "error",
        "email": {
            "enabled": True,
            "smtp_host": "smtp.test",
            "smtp_port": 587,
            "from_email": "alerts@test",
            "to_emails": ["ops@test"],
        },
        "webhooks": [
            {
                "url": "https://example.com/hook",
                "min_severity": "info",
                "headers": {"X-Test": "1"},
            }
        ],
    }

    dispatcher = alerts.AlertDispatcher.from_config(config)
    assert dispatcher.channels.keys() >= {"log", "slack", "pagerduty", "email", "webhook_0"}
    # Ensure constructor arguments flowed through
    channel_types = {name for name, *_ in added}
    assert {"slackstub", "pagerdutystub", "emailstub", "webhookstub"} <= channel_types


@pytest.mark.asyncio
async def test_slack_channel_sends_payload(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    payloads: list[dict[str, Any]] = []

    class DummyResponse:
        def __init__(self, status: int = 200) -> None:
            self.status = status

        async def __aenter__(self) -> DummyResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class DummySession:
        async def __aenter__(self) -> DummySession:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any], timeout: Any) -> DummyResponse:
            payloads.append({"url": url, "json": json})
            return DummyResponse()

    dummy_module = types.SimpleNamespace(
        ClientSession=lambda: DummySession(), ClientTimeout=lambda total: total
    )
    monkeypatch.setattr(alerts, "HAS_AIOHTTP", True)
    monkeypatch.setattr(alerts, "aiohttp", dummy_module, raising=False)

    channel = alerts.SlackChannel("https://hooks.slack.test/abc")
    sent = await channel.send(sample_alert)
    assert sent
    assert payloads and payloads[0]["url"].startswith("https://hooks.slack.test")


@pytest.mark.asyncio
async def test_webhook_channel_without_aiohttp(
    sample_alert: alerts.Alert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alerts, "HAS_AIOHTTP", False)
    channel = alerts.WebhookChannel("https://example.com/hook")
    assert not await channel.send(sample_alert)


@pytest.mark.asyncio
async def test_slack_channel_handles_failure(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    class DummyResponse:
        def __init__(self, status: int) -> None:
            self.status = status

        async def __aenter__(self) -> DummyResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class DummySession:
        async def __aenter__(self) -> DummySession:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any], timeout: Any) -> DummyResponse:
            return DummyResponse(status=500)

    dummy_module = types.SimpleNamespace(
        ClientSession=lambda: DummySession(), ClientTimeout=lambda total: total
    )
    monkeypatch.setattr(alerts, "HAS_AIOHTTP", True)
    monkeypatch.setattr(alerts, "aiohttp", dummy_module, raising=False)

    channel = alerts.SlackChannel("https://hooks.example/test")
    sent = await channel.send(sample_alert)
    assert not sent


@pytest.mark.asyncio
async def test_pagerduty_channel_sends_payload(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    requests: list[dict[str, Any]] = []

    class DummyResponse:
        status = 202

        async def __aenter__(self) -> DummyResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class DummySession:
        async def __aenter__(self) -> DummySession:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def post(
            self, url: str, *, json: dict[str, Any], headers: dict[str, Any], timeout: Any
        ) -> DummyResponse:
            requests.append({"url": url, "json": json, "headers": headers})
            return DummyResponse()

    dummy_module = types.SimpleNamespace(
        ClientSession=lambda: DummySession(), ClientTimeout=lambda total: total
    )
    monkeypatch.setattr(alerts, "HAS_AIOHTTP", True)
    monkeypatch.setattr(alerts, "aiohttp", dummy_module, raising=False)

    channel = alerts.PagerDutyChannel("routing-key", min_severity=alerts.AlertSeverity.ERROR)
    sent = await channel.send(sample_alert)
    assert sent
    assert requests
    payload = requests[0]["json"]
    assert payload["routing_key"] == "routing-key"
    assert payload["payload"]["severity"] == "error"


@pytest.mark.asyncio
async def test_email_channel_sends_message(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    sent_messages: list[dict[str, Any]] = []

    async def fake_send(msg: Any, **kwargs: Any) -> None:
        sent_messages.append({"subject": msg["Subject"], "kwargs": kwargs})

    monkeypatch.setattr(alerts, "HAS_AIOSMTPLIB", True)
    fake_module = types.SimpleNamespace(send=fake_send)
    monkeypatch.setitem(sys.modules, "aiosmtplib", fake_module)
    monkeypatch.setattr(alerts, "aiosmtplib", fake_module, raising=False)

    channel = alerts.EmailChannel(
        smtp_host="smtp.test",
        smtp_port=25,
        from_email="alerts@test",
        to_emails=["ops@test"],
        username="user",
        password="pass",
        use_tls=True,
        min_severity=alerts.AlertSeverity.ERROR,
    )

    sent = await channel.send(sample_alert)
    assert sent
    assert sent_messages
    assert "DISK FULL" in sent_messages[0]["subject"].upper()


@pytest.mark.asyncio
async def test_email_channel_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    monkeypatch.setattr(alerts, "HAS_AIOSMTPLIB", False)
    channel = alerts.EmailChannel(
        smtp_host="smtp.test",
        smtp_port=25,
        from_email="alerts@test",
        to_emails=["ops@test"],
    )
    sent = await channel.send(sample_alert)
    assert not sent


@pytest.mark.asyncio
async def test_pagerduty_channel_handles_failure(
    monkeypatch: pytest.MonkeyPatch, sample_alert: alerts.Alert
) -> None:
    class DummyResponse:
        status = 400

        async def __aenter__(self) -> DummyResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class DummySession:
        async def __aenter__(self) -> DummySession:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def post(
            self, url: str, *, json: dict[str, Any], headers: dict[str, Any], timeout: Any
        ) -> DummyResponse:
            return DummyResponse()

    dummy_module = types.SimpleNamespace(
        ClientSession=lambda: DummySession(), ClientTimeout=lambda total: total
    )
    monkeypatch.setattr(alerts, "HAS_AIOHTTP", True)
    monkeypatch.setattr(alerts, "aiohttp", dummy_module, raising=False)

    channel = alerts.PagerDutyChannel("routing", min_severity=alerts.AlertSeverity.ERROR)
    sent = await channel.send(sample_alert)
    assert not sent


@pytest.mark.asyncio
async def test_email_channel_send_failure_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, sample_alert: alerts.Alert
) -> None:
    async def failing_send(msg: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise RuntimeError("smtp down")

    monkeypatch.setattr(alerts, "HAS_AIOSMTPLIB", True)
    fake_module = types.SimpleNamespace(send=failing_send)
    monkeypatch.setitem(sys.modules, "aiosmtplib", fake_module)
    monkeypatch.setattr(alerts, "aiosmtplib", fake_module, raising=False)

    caplog.set_level("ERROR")
    channel = alerts.EmailChannel(
        smtp_host="smtp.test",
        smtp_port=25,
        from_email="alerts@test",
        to_emails=["ops@test"],
    )

    sent = await channel.send(sample_alert)
    assert not sent
    assert any("Failed to send email" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_log_channel_emits(
    caplog: pytest.LogCaptureFixture, sample_alert: alerts.Alert
) -> None:
    caplog.set_level("INFO")
    channel = alerts.LogChannel(min_severity=alerts.AlertSeverity.DEBUG)
    sent = await channel.send(sample_alert)
    assert sent
    assert any("Disk Full" in record.message for record in caplog.records)


def test_convenience_alert_builders() -> None:
    risk = alerts.create_risk_alert("Breach", "Position over leverage")
    execution = alerts.create_execution_alert("Fill", "Order filled")
    system = alerts.create_system_alert("Down", "Service unavailable")

    assert risk.source == "risk_manager"
    assert execution.source == "execution_engine"
    assert system.source == "system"
