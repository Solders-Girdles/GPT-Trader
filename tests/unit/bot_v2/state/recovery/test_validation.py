"""Tests for the recovery validation pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_v2.state.recovery import validation
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)


@pytest.fixture
def state_manager() -> MagicMock:
    """Provide a mock state manager with async helpers."""
    manager = MagicMock()
    manager.get_state = AsyncMock(
        return_value={
            "positions": [],
            "cash_balance": 1_000,
            "total_value": 1_000,
        }
    )
    return manager


@pytest.fixture
def checkpoint_handler() -> object:
    """Opaque checkpoint handler dependency."""
    return object()


@pytest.fixture
def recovery_operation() -> RecoveryOperation:
    """Create a recovery operation to validate."""
    event = FailureEvent(
        failure_type=FailureType.REDIS_DOWN,
        timestamp=datetime.utcnow(),
        severity="high",
        affected_components=["redis"],
        error_message="Redis ping timeout",
    )
    return RecoveryOperation(
        operation_id="op-456",
        failure_event=event,
        recovery_mode=RecoveryMode.AUTOMATIC,
        status=RecoveryStatus.VALIDATING,
        started_at=datetime.utcnow(),
    )


@pytest.fixture
def detector_stub(monkeypatch) -> SimpleNamespace:
    """Patch FailureDetector with async-aware stub methods."""
    stub = SimpleNamespace(
        test_redis_health=AsyncMock(return_value=True),
        test_postgres_health=AsyncMock(return_value=True),
        detect_data_corruption=AsyncMock(return_value=False),
        test_trading_engine_health=AsyncMock(return_value=True),
    )

    def factory(state_manager, checkpoint_handler):  # type: ignore[unused-argument]
        return stub

    monkeypatch.setattr(validation, "FailureDetector", factory)
    return stub


@pytest.mark.asyncio
async def test_validate_recovery_success_path(
    state_manager, checkpoint_handler, recovery_operation, detector_stub
) -> None:
    """All checks passing should mark critical validations as successful."""
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_recovery(recovery_operation)

    assert result is True
    assert detector_stub.test_redis_health.await_count == 1
    assert recovery_operation.validation_results["redis_health"] is True
    assert recovery_operation.validation_results["critical_data"] is True
    assert recovery_operation.actions_taken[-1].startswith("Validation completed in")


@pytest.mark.asyncio
async def test_validate_recovery_flags_failed_critical_checks(
    state_manager, checkpoint_handler, recovery_operation, detector_stub, caplog
) -> None:
    """Critical validation failure should log a warning and return False."""
    detector_stub.detect_data_corruption.return_value = True
    caplog.set_level(logging.WARNING)
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_recovery(recovery_operation)

    assert result is False
    assert "Recovery validation failed" in caplog.text
    assert recovery_operation.validation_results["data_integrity"] is False


@pytest.mark.asyncio
async def test_validate_recovery_handles_detector_errors(
    state_manager, checkpoint_handler, recovery_operation, detector_stub, caplog
) -> None:
    """Detector exceptions should be caught and reported as validation failure."""
    detector_stub.test_redis_health.side_effect = RuntimeError("redis offline")
    caplog.set_level(logging.ERROR)
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_recovery(recovery_operation)

    assert result is False
    assert "Recovery validation error: redis offline" in caplog.text
    assert recovery_operation.validation_results == {}


@pytest.mark.asyncio
async def test_validate_critical_data_success(
    state_manager, checkpoint_handler, detector_stub
) -> None:
    """Healthy portfolio data should pass validation."""
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    assert await validator.validate_critical_data() is True
    state_manager.get_state.assert_awaited_once_with("portfolio_current")


@pytest.mark.asyncio
async def test_validate_critical_data_missing_portfolio(
    state_manager, checkpoint_handler, detector_stub, caplog
) -> None:
    """Missing portfolio data is treated as a validation failure."""
    state_manager.get_state.return_value = None
    caplog.set_level(logging.WARNING)
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_critical_data()

    assert result is False
    assert "Portfolio data missing" in caplog.text


@pytest.mark.asyncio
async def test_validate_critical_data_missing_field(
    state_manager, checkpoint_handler, detector_stub, caplog
) -> None:
    """Incomplete portfolio data should fail validation with logged warning."""
    state_manager.get_state.return_value = {"positions": [], "total_value": 1_000}
    caplog.set_level(logging.WARNING)
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_critical_data()

    assert result is False
    assert "Portfolio missing field: cash_balance" in caplog.text


@pytest.mark.asyncio
async def test_validate_critical_data_handles_exceptions(
    state_manager, checkpoint_handler, detector_stub, caplog
) -> None:
    """Unexpected errors should be caught and surfaced as validation failure."""
    state_manager.get_state.side_effect = RuntimeError("redis timeout")
    caplog.set_level(logging.ERROR)
    validator = validation.RecoveryValidator(state_manager, checkpoint_handler)

    result = await validator.validate_critical_data()

    assert result is False
    assert "Critical data validation error: redis timeout" in caplog.text


class TestValidatePosition:
    """Unit tests for position validation helper."""

    def test_position_with_all_fields_is_valid(
        self, state_manager, checkpoint_handler, detector_stub
    ) -> None:
        validator = validation.RecoveryValidator(state_manager, checkpoint_handler)
        position = {"symbol": "AAPL", "quantity": 10, "entry_price": 100}
        assert validator.validate_position(position) is True

    def test_position_with_missing_field_is_invalid(
        self, state_manager, checkpoint_handler, detector_stub
    ) -> None:
        validator = validation.RecoveryValidator(state_manager, checkpoint_handler)
        assert validator.validate_position({"symbol": "AAPL", "quantity": 10}) is False
