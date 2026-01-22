"""Tests for staleness helper age, severity, label, and display helpers."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.staleness_helpers import (
    format_freshness_label,
    get_data_age_seconds,
    get_freshness_display,
    get_staleness_severity,
)


def make_state(**attrs: object) -> MagicMock:
    state = MagicMock()
    for key, value in attrs.items():
        setattr(state, key, value)
    return state


@pytest.mark.parametrize(
    ("data_available", "last_data_fetch"),
    [
        # Use a deterministic value; pytest-xdist requires stable collection across workers.
        (False, 1_700_000_000.0),
        (True, 0),
        (True, -1),
    ],
)
def test_get_data_age_seconds_returns_none(data_available, last_data_fetch) -> None:
    state = make_state(data_available=data_available, last_data_fetch=last_data_fetch)

    assert get_data_age_seconds(state) is None


def test_get_data_age_seconds_returns_age_when_available() -> None:
    state = make_state(data_available=True, last_data_fetch=time.time() - 15)

    age = get_data_age_seconds(state)

    assert age is not None
    assert 14 <= age <= 16


@pytest.mark.parametrize(
    ("connection_healthy", "data_available", "last_fetch_age", "expected"),
    [
        (False, True, 1, "critical"),
        (True, False, None, "fresh"),
        (True, True, 5, "fresh"),
        (True, True, 10, "stale"),
        (True, True, 20, "stale"),
        (True, True, 30, "critical"),
        (True, True, 45, "critical"),
    ],
)
def test_get_staleness_severity_cases(
    connection_healthy,
    data_available,
    last_fetch_age,
    expected,
) -> None:
    last_data_fetch = 0 if last_fetch_age is None else time.time() - last_fetch_age
    state = make_state(
        connection_healthy=connection_healthy,
        data_available=data_available,
        last_data_fetch=last_data_fetch,
    )

    assert get_staleness_severity(state) == expected


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (5, "5s ago"),
        (45, "45s ago"),
        (120, "2m ago"),
        (3000, "50m ago"),
        (3600, "1h ago"),
        (7200, "2h ago"),
    ],
)
def test_format_freshness_label(age, expected) -> None:
    assert format_freshness_label(age) == expected


def test_get_freshness_display_returns_syncing_when_fetching() -> None:
    state = make_state(data_fetching=True)

    result = get_freshness_display(state)

    assert result is not None
    text, css_class = result
    assert "syncing" in text
    assert css_class == "fresh"


def test_get_freshness_display_returns_none_when_no_data() -> None:
    state = make_state(
        data_fetching=False,
        data_available=False,
        last_data_fetch=0,
    )

    assert get_freshness_display(state) is None


@pytest.mark.parametrize(
    ("last_fetch_age", "connection_healthy", "tag", "css_class"),
    [
        (5, True, "[dim]", "fresh"),
        (20, True, "[yellow]", "stale"),
        (45, True, "[red]", "critical"),
        (5, False, "[red]", "critical"),
    ],
)
def test_get_freshness_display_styles(last_fetch_age, connection_healthy, tag, css_class) -> None:
    state = make_state(
        data_fetching=False,
        data_available=True,
        connection_healthy=connection_healthy,
        last_data_fetch=time.time() - last_fetch_age,
    )

    result = get_freshness_display(state)

    assert result is not None
    text, style = result
    assert tag in text
    assert style == css_class
