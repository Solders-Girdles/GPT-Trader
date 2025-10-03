from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from bot_v2.monitoring.domain.perps import margin


class FrozenDateTime(datetime):
    """Helper to control utcnow/now in margin module during tests."""

    frozen: datetime = datetime(2025, 1, 15, 12, 0, 0)

    @classmethod
    def utcnow(cls) -> datetime:
        return cls.frozen

    @classmethod
    def now(cls) -> datetime:
        return cls.frozen


@pytest.fixture(autouse=True)
def _patch_datetime(monkeypatch: pytest.MonkeyPatch) -> FrozenDateTime:
    monkeypatch.setattr(margin, "datetime", FrozenDateTime)
    return FrozenDateTime


def test_margin_window_policy_determines_windows() -> None:
    policy = margin.MarginWindowPolicy()

    assert (
        policy.determine_current_window(datetime(2025, 1, 15, 0, 0))
        is margin.MarginWindow.PRE_FUNDING
    )
    # Wrap-around branch (23:45 -> 00:00 funding)
    assert (
        policy.determine_current_window(datetime(2025, 1, 14, 23, 45))
        is margin.MarginWindow.PRE_FUNDING
    )
    assert (
        policy.determine_current_window(datetime(2025, 1, 15, 22, 1))
        is margin.MarginWindow.OVERNIGHT
    )
    assert (
        policy.determine_current_window(datetime(2025, 1, 15, 14, 30))
        is margin.MarginWindow.INTRADAY
    )
    assert (
        policy.determine_current_window(datetime(2025, 1, 15, 11, 0)) is margin.MarginWindow.NORMAL
    )


def test_margin_window_policy_next_change() -> None:
    policy = margin.MarginWindowPolicy()
    current_time = datetime(2025, 1, 15, 7, 45)
    next_change = policy.calculate_next_window_change(current_time)
    assert next_change.hour == 8
    assert next_change.minute == 0
    assert next_change.date() == current_time.date()


def test_margin_window_policy_risk_reduction() -> None:
    policy = margin.MarginWindowPolicy()
    assert policy.should_reduce_risk(margin.MarginWindow.NORMAL, margin.MarginWindow.PRE_FUNDING)
    assert not policy.should_reduce_risk(
        margin.MarginWindow.OVERNIGHT, margin.MarginWindow.OVERNIGHT
    )


@pytest.mark.asyncio
async def test_margin_state_monitor_compute_and_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    async def fake_transition() -> None | dict[str, Any]:
        return None

    monitor = margin.MarginStateMonitor(alert_threshold=Decimal("0.75"))
    monkeypatch.setattr(monitor, "check_window_transition", fake_transition)

    async def callback(snapshot: margin.MarginSnapshot, alert_type: str) -> None:
        captured.append(alert_type)

    monitor.add_alert_callback(callback)

    positions = {
        "BTC-USD-PERP": {
            "quantity": "8",
            "mark_price": "100",
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=Decimal("100"),
        cash_balance=Decimal("25"),
        positions=positions,
    )

    assert snapshot.margin_utilization == Decimal("0.80")
    assert snapshot.margin_available == Decimal("20")
    assert snapshot.leverage == Decimal("8")
    assert captured == ["HIGH_UTILIZATION"]
    assert monitor.get_current_state() is snapshot


@pytest.mark.asyncio
async def test_margin_state_monitor_risk_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_transition() -> dict[str, Any] | None:
        return {
            "current_window": margin.MarginWindow.NORMAL.value,
            "next_window": margin.MarginWindow.PRE_FUNDING.value,
            "change_time": "2025-01-15T12:30:00",
            "minutes_until": 30,
            "should_reduce_risk": True,
            "current_max_leverage": 10.0,
            "next_max_leverage": 4.0,
            "leverage_reduction_factor": 0.4,
        }

    monitor = margin.MarginStateMonitor(
        alert_threshold=Decimal("0.50"), liquidation_buffer=Decimal("0.20")
    )
    monkeypatch.setattr(monitor, "check_window_transition", fake_transition)

    alerts: list[str] = []

    async def callback(snapshot: margin.MarginSnapshot, alert_type: str) -> None:
        alerts.append(alert_type)

    monitor.add_alert_callback(callback)

    positions = {
        "ETH-USD-PERP": {
            "quantity": "5",
            "mark_price": "100",
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=Decimal("20"),
        cash_balance=Decimal("5"),
        positions=positions,
    )

    assert snapshot.is_margin_call
    assert snapshot.is_liquidation_risk
    assert "LIQUIDATION_RISK" in alerts
    assert "WINDOW_TRANSITION" in alerts


@pytest.mark.asyncio
async def test_margin_state_monitor_history_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = margin.MarginStateMonitor()

    async def fake_transition() -> None:
        return None

    monkeypatch.setattr(monitor, "check_window_transition", fake_transition)

    positions = {
        "BTC-USD-PERP": {
            "quantity": "1",
            "mark_price": "100",
        }
    }

    # Record multiple snapshots to exercise history trimming
    for minutes in range(0, 600, 5):
        FrozenDateTime.frozen = datetime(2025, 1, 15, 12, 0) + timedelta(minutes=minutes)
        await monitor.compute_margin_state(
            total_equity=Decimal("200"),
            cash_balance=Decimal("150"),
            positions=positions,
        )

    history = monitor.get_margin_history(hours_back=2)
    assert history
    assert all(entry["timestamp"].startswith("2025-01-15T") for entry in history)
    assert monitor.get_current_window() in margin.MarginWindow

    FrozenDateTime.frozen = datetime(2025, 1, 15, 12, 0)
    size = await monitor.get_max_position_size(symbol="BTC-USD-PERP", price=Decimal("100"))
    assert size > 0


@pytest.mark.asyncio
async def test_check_window_transition_detects_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = margin.MarginStateMonitor()

    async def fake_collect(*_: Any) -> None:
        return None

    monkeypatch.setattr(monitor, "_alert_callbacks", [])

    FrozenDateTime.frozen = datetime(2025, 1, 15, 21, 40)
    transition = await monitor.check_window_transition()
    assert transition is not None
    assert transition["current_window"] == margin.MarginWindow.NORMAL.value
    assert transition["next_window"] == margin.MarginWindow.OVERNIGHT.value
    assert transition["should_reduce_risk"]


@pytest.mark.asyncio
async def test_create_margin_monitor_utility() -> None:
    monitor = await margin.create_margin_monitor()
    assert isinstance(monitor, margin.MarginStateMonitor)
