"""Edge-case tests for StateValidator."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from gpt_trader.monitoring.configuration_guardian.models import BaselineSnapshot
from gpt_trader.monitoring.configuration_guardian.responses import DriftResponse
from gpt_trader.monitoring.configuration_guardian.state_validator import StateValidator


def test_calculate_current_leverage_zero_when_equity_missing_or_nonpositive() -> None:
    baseline = BaselineSnapshot(active_symbols=["BTC-USD"], profile="test")
    validator = StateValidator(baseline)
    positions = [SimpleNamespace(size=1, price=100)]

    assert validator._calculate_current_leverage(positions, None) == Decimal("0")
    assert validator._calculate_current_leverage(positions, Decimal("0")) == Decimal("0")
    assert validator._calculate_current_leverage(positions, Decimal("-1")) == Decimal("0")


def test_remove_symbol_with_active_position_emits_event() -> None:
    baseline = BaselineSnapshot(active_symbols=["BTC-USD", "ETH-USD"], profile="test")
    validator = StateValidator(baseline)
    positions = [SimpleNamespace(symbol="BTC-USD", size=1, price=100)]

    events = validator.validate_config_against_state(
        {
            "symbols": ["ETH-USD"],
            "max_leverage": 10,
            "max_position_size": Decimal("10000"),
            "profile": "test",
        },
        [],
        positions,
        Decimal("1000"),
    )

    event = next(evt for evt in events if evt.drift_type == "symbols_remove_active_positions")
    assert event.severity == "critical"
    assert event.suggested_response == DriftResponse.EMERGENCY_SHUTDOWN
    assert event.applied_response == DriftResponse.EMERGENCY_SHUTDOWN


def test_leverage_violation_emits_event() -> None:
    baseline = BaselineSnapshot(active_symbols=["BTC-USD"], profile="test")
    validator = StateValidator(baseline)
    positions = [SimpleNamespace(symbol="BTC-USD", size=2, price=100)]

    events = validator.validate_config_against_state(
        {"symbols": ["BTC-USD"], "max_leverage": 1, "profile": "test"},
        [],
        positions,
        Decimal("100"),
    )

    event = next(evt for evt in events if evt.drift_type == "leverage_violation_current_positions")
    assert event.severity == "high"
    assert event.suggested_response == DriftResponse.REDUCE_ONLY
    assert event.applied_response == DriftResponse.REDUCE_ONLY


def test_position_size_violation_emits_event() -> None:
    baseline = BaselineSnapshot(active_symbols=["BTC-USD"], profile="test")
    validator = StateValidator(baseline)
    positions = [SimpleNamespace(symbol="BTC-USD", size=2, price=10)]

    events = validator.validate_config_against_state(
        {
            "symbols": ["BTC-USD"],
            "max_position_size": Decimal("10"),
            "profile": "test",
        },
        [],
        positions,
        Decimal("1000"),
    )

    event = next(
        evt for evt in events if evt.drift_type == "position_size_violation_current_exposure"
    )
    assert event.severity == "high"
    assert event.suggested_response == DriftResponse.REDUCE_ONLY
    assert event.applied_response == DriftResponse.REDUCE_ONLY


def test_profile_change_emits_event() -> None:
    baseline = BaselineSnapshot(active_symbols=["BTC-USD"], profile="alpha")
    validator = StateValidator(baseline)

    events = validator.validate_config_against_state(
        {"symbols": ["BTC-USD"], "profile": "beta"},
        [],
        [],
        Decimal("1000"),
    )

    event = next(evt for evt in events if evt.drift_type == "profile_changed_during_runtime")
    assert event.severity == "critical"
    assert event.suggested_response == DriftResponse.EMERGENCY_SHUTDOWN
    assert event.applied_response == DriftResponse.EMERGENCY_SHUTDOWN
