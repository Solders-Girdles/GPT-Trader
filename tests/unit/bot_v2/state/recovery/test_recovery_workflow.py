"""
Unit tests for RecoveryWorkflow

Tests workflow execution paths, validation integration, alerting, and escalation logic.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.state.recovery.alerting import RecoveryAlerter
from bot_v2.state.recovery.handler_registry import RecoveryHandlerRegistry
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)
from bot_v2.state.recovery.validation import RecoveryValidator
from bot_v2.state.recovery.workflow import RecoveryOutcome, RecoveryWorkflow


@pytest.fixture
def mock_handler_registry():
    """Create mock handler registry."""
    return Mock(spec=RecoveryHandlerRegistry)


@pytest.fixture
def mock_validator():
    """Create mock validator."""
    validator = Mock(spec=RecoveryValidator)
    validator.validate_recovery = AsyncMock(return_value=True)
    return validator


@pytest.fixture
def mock_alerter():
    """Create mock alerter."""
    alerter = Mock(spec=RecoveryAlerter)
    alerter.send_alert = AsyncMock()
    alerter.escalate_recovery = AsyncMock()
    return alerter


@pytest.fixture
def recovery_config():
    """Create recovery config."""
    return RecoveryConfig(
        rto_minutes=5,
        max_retry_attempts=3,
        retry_delay_seconds=0.001,  # Fast for tests
    )


@pytest.fixture
def workflow(mock_handler_registry, mock_validator, mock_alerter, recovery_config):
    """Create workflow instance."""
    return RecoveryWorkflow(
        handler_registry=mock_handler_registry,
        validator=mock_validator,
        alerter=mock_alerter,
        config=recovery_config,
    )


@pytest.fixture
def sample_operation():
    """Create sample recovery operation."""
    return RecoveryOperation(
        operation_id="TEST_001",
        failure_event=FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="critical",
            affected_components=["cache"],
            error_message="Redis connection lost",
        ),
        recovery_mode=RecoveryMode.AUTOMATIC,
        status=RecoveryStatus.IN_PROGRESS,
        started_at=datetime.utcnow(),
    )


class TestWorkflowInit:
    """Test workflow initialization."""

    def test_init_with_all_dependencies(
        self, mock_handler_registry, mock_validator, mock_alerter, recovery_config
    ):
        """Should initialize with all dependencies."""
        workflow = RecoveryWorkflow(
            handler_registry=mock_handler_registry,
            validator=mock_validator,
            alerter=mock_alerter,
            config=recovery_config,
        )

        assert workflow.handler_registry is mock_handler_registry
        assert workflow.validator is mock_validator
        assert workflow.alerter is mock_alerter
        assert workflow.config is recovery_config

    def test_init_with_default_config(self, mock_handler_registry, mock_validator, mock_alerter):
        """Should use default config if not provided."""
        workflow = RecoveryWorkflow(
            handler_registry=mock_handler_registry,
            validator=mock_validator,
            alerter=mock_alerter,
        )

        assert workflow.config is not None
        assert isinstance(workflow.config, RecoveryConfig)


class TestSuccessPath:
    """Test successful recovery workflow."""

    @pytest.mark.asyncio
    async def test_execute_success_with_validation(
        self, workflow, sample_operation, mock_handler_registry, mock_validator, mock_alerter
    ):
        """Should complete successfully when handler and validation pass."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler.__name__ = "test_handler"
        mock_handler_registry.get_handler.return_value = mock_handler
        mock_validator.validate_recovery.return_value = True

        # Execute
        outcome = await workflow.execute(sample_operation, RecoveryMode.AUTOMATIC)

        # Verify outcome
        assert outcome.success is True
        assert outcome.status == RecoveryStatus.COMPLETED
        assert outcome.validation_passed is True
        assert outcome.escalation_required is False
        assert outcome.recovery_time_seconds is not None
        assert len(outcome.actions_taken) > 0

        # Verify handler called
        mock_handler.assert_called_once_with(sample_operation)

        # Verify validation called
        mock_validator.validate_recovery.assert_called_once_with(sample_operation)

        # Verify alerts sent
        assert mock_alerter.send_alert.call_count == 2  # Start and completion
        mock_alerter.escalate_recovery.assert_not_called()

        # Verify operation updated
        assert sample_operation.status == RecoveryStatus.COMPLETED
        assert sample_operation.completed_at is not None
        assert sample_operation.recovery_time_seconds is not None

    @pytest.mark.asyncio
    async def test_execute_updates_operation_timestamps(
        self, workflow, sample_operation, mock_handler_registry
    ):
        """Should set completion timestamp and calculate recovery time."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler
        start_time = datetime.utcnow()
        sample_operation.started_at = start_time

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify
        assert sample_operation.completed_at is not None
        assert sample_operation.completed_at >= start_time
        assert outcome.recovery_time_seconds is not None
        assert outcome.recovery_time_seconds >= 0


class TestValidationFailure:
    """Test validation failure scenarios."""

    @pytest.mark.asyncio
    async def test_execute_partial_when_validation_fails(
        self, workflow, sample_operation, mock_handler_registry, mock_validator
    ):
        """Should return partial status when handler succeeds but validation fails."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler
        mock_validator.validate_recovery.return_value = False

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify outcome
        assert outcome.success is False
        assert outcome.status == RecoveryStatus.PARTIAL
        assert outcome.validation_passed is False
        assert outcome.escalation_required is False
        assert outcome.error_message == "Validation failed after handler execution"

        # Verify operation status
        assert sample_operation.status == RecoveryStatus.PARTIAL


class TestHandlerFailure:
    """Test handler failure scenarios."""

    @pytest.mark.asyncio
    async def test_execute_fails_when_no_handler_registered(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should fail gracefully when no handler registered."""
        # Setup
        mock_handler_registry.get_handler.return_value = None

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify outcome
        assert outcome.success is False
        assert outcome.status == RecoveryStatus.FAILED
        assert outcome.escalation_required is True  # Automatic mode
        assert len(outcome.actions_taken) > 0
        assert "No handler found" in outcome.actions_taken[0]

        # Verify escalation called
        mock_alerter.escalate_recovery.assert_called_once_with(sample_operation)

    @pytest.mark.asyncio
    async def test_execute_retries_handler_on_failure(
        self, workflow, sample_operation, mock_handler_registry, recovery_config
    ):
        """Should retry handler on failure."""
        # Setup - fail twice, succeed on third
        mock_handler = AsyncMock(side_effect=[False, False, True])
        mock_handler.__name__ = "test_handler"
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify retries
        assert mock_handler.call_count == 3
        assert outcome.success is True

    @pytest.mark.asyncio
    async def test_execute_fails_after_all_retries_exhausted(
        self, workflow, sample_operation, mock_handler_registry, recovery_config
    ):
        """Should fail when all retry attempts exhausted."""
        # Setup - always return False
        mock_handler = AsyncMock(return_value=False)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify
        assert mock_handler.call_count == recovery_config.max_retry_attempts
        assert outcome.success is False
        assert outcome.status == RecoveryStatus.FAILED
        assert outcome.escalation_required is True

    @pytest.mark.asyncio
    async def test_execute_captures_handler_exceptions(
        self, workflow, sample_operation, mock_handler_registry, recovery_config
    ):
        """Should capture and log handler exceptions."""
        # Setup
        mock_handler = AsyncMock(side_effect=RuntimeError("Handler exploded"))
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify
        assert outcome.success is False
        assert mock_handler.call_count == recovery_config.max_retry_attempts
        assert any("Handler error" in action for action in outcome.actions_taken)


class TestEscalation:
    """Test escalation logic."""

    @pytest.mark.asyncio
    async def test_escalation_required_for_automatic_mode_failure(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should escalate when automatic mode fails."""
        # Setup
        mock_handler = AsyncMock(return_value=False)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute in automatic mode
        outcome = await workflow.execute(sample_operation, RecoveryMode.AUTOMATIC)

        # Verify escalation
        assert outcome.escalation_required is True
        mock_alerter.escalate_recovery.assert_called_once_with(sample_operation)

    @pytest.mark.asyncio
    async def test_no_escalation_for_manual_mode_failure(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should not escalate when manual mode fails."""
        # Setup
        mock_handler = AsyncMock(return_value=False)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute in manual mode
        outcome = await workflow.execute(sample_operation, RecoveryMode.MANUAL)

        # Verify no escalation
        assert outcome.escalation_required is False
        mock_alerter.escalate_recovery.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_escalation_on_success(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should not escalate on successful recovery."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation, RecoveryMode.AUTOMATIC)

        # Verify no escalation
        assert outcome.escalation_required is False
        mock_alerter.escalate_recovery.assert_not_called()


class TestAlerting:
    """Test alert dispatch."""

    @pytest.mark.asyncio
    async def test_alerts_sent_on_success(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should send start and completion alerts on success."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        await workflow.execute(sample_operation)

        # Verify alerts
        assert mock_alerter.send_alert.call_count == 2
        calls = mock_alerter.send_alert.call_args_list

        # First call: recovery started
        assert "started" in calls[0][0][0].lower()

        # Second call: recovery completed
        assert "completed" in calls[1][0][0].lower()

    @pytest.mark.asyncio
    async def test_alerts_sent_on_failure(
        self, workflow, sample_operation, mock_handler_registry, mock_alerter
    ):
        """Should send start and failure alerts on failure."""
        # Setup
        mock_handler = AsyncMock(return_value=False)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        await workflow.execute(sample_operation)

        # Verify alerts
        assert mock_alerter.send_alert.call_count == 2
        calls = mock_alerter.send_alert.call_args_list

        # First call: recovery started
        assert "started" in calls[0][0][0].lower()

        # Second call: recovery failed
        assert "failed" in calls[1][0][0].lower()


class TestRTOCompliance:
    """Test RTO compliance checking."""

    @pytest.mark.asyncio
    async def test_rto_exceeded_flag_when_recovery_slow(
        self, workflow, sample_operation, mock_handler_registry, recovery_config
    ):
        """Should set rto_exceeded flag when recovery time exceeds RTO."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Simulate slow recovery (manipulate timestamps)
        sample_operation.started_at = datetime.utcnow() - timedelta(minutes=10)

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify RTO exceeded
        assert outcome.rto_exceeded is True
        assert outcome.recovery_time_seconds > recovery_config.rto_minutes * 60

    @pytest.mark.asyncio
    async def test_rto_not_exceeded_when_recovery_fast(
        self, workflow, sample_operation, mock_handler_registry, recovery_config
    ):
        """Should not set rto_exceeded flag when recovery time within RTO."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute (fast recovery)
        outcome = await workflow.execute(sample_operation)

        # Verify RTO not exceeded
        assert outcome.rto_exceeded is False
        assert outcome.recovery_time_seconds <= recovery_config.rto_minutes * 60


class TestActionsTaken:
    """Test actions tracking."""

    @pytest.mark.asyncio
    async def test_actions_taken_includes_handler_success(
        self, workflow, sample_operation, mock_handler_registry
    ):
        """Should record handler success in actions_taken."""
        # Setup
        mock_handler = AsyncMock(return_value=True)
        mock_handler.__name__ = "my_handler"
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify
        assert any("my_handler" in action for action in outcome.actions_taken)
        assert any("Successfully executed" in action for action in outcome.actions_taken)

    @pytest.mark.asyncio
    async def test_actions_taken_includes_handler_errors(
        self, workflow, sample_operation, mock_handler_registry
    ):
        """Should record handler errors in actions_taken."""
        # Setup
        mock_handler = AsyncMock(side_effect=ValueError("Test error"))
        mock_handler_registry.get_handler.return_value = mock_handler

        # Execute
        outcome = await workflow.execute(sample_operation)

        # Verify
        assert any("Handler error" in action for action in outcome.actions_taken)
        assert any("Test error" in action for action in outcome.actions_taken)
