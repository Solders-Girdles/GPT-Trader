"""Tests for staleness banners and empty-state helpers."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.staleness_helpers import (
    get_connection_banner,
    get_empty_state_config,
    get_staleness_banner,
)


def make_state(**attrs: object) -> MagicMock:
    state = MagicMock()
    state.data_fetching = False
    state.degraded_mode = False
    state.degraded_reason = "Test reason"
    state.data_available = True
    state.connection_healthy = True
    state.last_data_fetch = time.time() - 5
    state.resilience_data.any_circuit_open = False
    state.execution_data.submissions_total = 0
    for key, value in attrs.items():
        setattr(state, key, value)
    return state


def test_get_staleness_banner_returns_reconnecting_when_fetching() -> None:
    state = make_state(data_fetching=True)

    result = get_staleness_banner(state)

    assert result is not None
    text, severity = result
    assert "Reconnecting" in text
    assert severity == "info"


def test_get_staleness_banner_returns_degraded_banner() -> None:
    state = make_state(degraded_mode=True, degraded_reason="Test reason")

    result = get_staleness_banner(state)

    assert result is not None
    text, severity = result
    assert "Degraded" in text
    assert "Test reason" in text
    assert severity == "warning"


def test_get_staleness_banner_handles_none_degraded_reason() -> None:
    state = make_state(degraded_mode=True, degraded_reason=None)

    result = get_staleness_banner(state)

    assert result is not None
    text, severity = result
    assert "Degraded" in text
    assert "Status reporter unavailable" in text
    assert severity == "warning"


@pytest.mark.parametrize(
    ("last_fetch_age", "expected_severity", "expected_hint"),
    [
        (5, None, None),
        (20, "warning", None),
        (45, "error", "R"),
    ],
)
def test_get_staleness_banner_staleness_levels(
    last_fetch_age,
    expected_severity,
    expected_hint,
) -> None:
    state = make_state(last_data_fetch=time.time() - last_fetch_age)

    result = get_staleness_banner(state)

    if expected_severity is None:
        assert result is None
        return

    assert result is not None
    text, severity = result
    assert "stale" in text.lower()
    assert severity == expected_severity
    if expected_hint:
        assert expected_hint in text


def test_get_staleness_banner_connection_unhealthy_with_fresh_data() -> None:
    state = make_state(connection_healthy=False, last_data_fetch=time.time() - 5)

    result = get_staleness_banner(state)

    assert result is not None
    text, severity = result
    assert "stale" in text.lower()
    assert "R" in text
    assert severity == "error"


def test_get_staleness_banner_returns_none_when_no_data_yet() -> None:
    state = make_state(data_available=False, last_data_fetch=0)

    assert get_staleness_banner(state) is None


def test_get_connection_banner_returns_none_when_stopped() -> None:
    assert (
        get_connection_banner(
            connection_status="UNKNOWN",
            bot_running=False,
            degraded_mode=False,
        )
        is None
    )


@pytest.mark.parametrize("status", ["CONNECTING", "RECONNECTING", "SYNCING", "UNKNOWN", "--", ""])
def test_get_connection_banner_returns_none_for_transitional_states(status) -> None:
    result = get_connection_banner(
        connection_status=status,
        bot_running=True,
        degraded_mode=False,
    )

    assert result is None


@pytest.mark.parametrize("status", ["CONNECTED", "OK", "HEALTHY"])
def test_get_connection_banner_returns_none_for_healthy_states(status) -> None:
    result = get_connection_banner(
        connection_status=status,
        bot_running=True,
        degraded_mode=False,
    )

    assert result is None


@pytest.mark.parametrize("status", ["DISCONNECTED", "disconnected"])
def test_get_connection_banner_returns_error_for_disconnected(status) -> None:
    result = get_connection_banner(
        connection_status=status,
        bot_running=True,
        degraded_mode=False,
    )

    assert result is not None
    text, severity = result
    assert "Connection lost" in text
    assert severity == "error"


def test_get_empty_state_config_stopped_when_not_running() -> None:
    config = get_empty_state_config(
        data_type="Market",
        bot_running=False,
        data_source_mode="paper",
        connection_status="",
    )
    assert "Awaiting Start" in config["title"]
    assert "[S] Start Bot" in config["actions"]


def test_get_empty_state_config_read_only() -> None:
    config = get_empty_state_config(
        data_type="Market",
        bot_running=False,
        data_source_mode="read_only",
        connection_status="",
    )
    assert "Feed Paused" in config["title"]
    assert "[S] Start Feed" in config["actions"]


def test_get_empty_state_config_connection_failed() -> None:
    config = get_empty_state_config(
        data_type="Market",
        bot_running=True,
        data_source_mode="live",
        connection_status="DISCONNECTED",
    )
    assert "Connection Failed" in config["title"]
    assert "[R] Reconnect" in config["actions"]


@pytest.mark.parametrize("status", ["CONNECTING", None])
def test_get_empty_state_config_connecting(status) -> None:
    config = get_empty_state_config(
        data_type="Market",
        bot_running=True,
        data_source_mode="live",
        connection_status=status,  # type: ignore[arg-type]
    )
    assert "Connecting" in config["title"]
    assert config["actions"] == []


def test_get_empty_state_config_no_data() -> None:
    config = get_empty_state_config(
        data_type="Strategy",
        bot_running=True,
        data_source_mode="live",
        connection_status="CONNECTED",
    )
    assert "No Strategy Yet" in config["title"]
    assert "[R] Refresh" in config["actions"]
