"""Tests for recovery orchestrator"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.state.recovery.orchestrator import RecoveryOrchestrator, detect_and_recover
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
    RecoveryStatus,
)


@pytest.fixture
def mock_state_manager():
    """Mock state manager"""
    manager = Mock()
    manager.redis_client = Mock()
    manager.pg_conn = Mock()
    manager.s3_client = Mock()
    manager.get_state = AsyncMock()
    manager.set_state = AsyncMock()
    return manager


@pytest.fixture
def mock_checkpoint_handler():
    """Mock checkpoint handler"""
    return Mock()


@pytest.fixture
def mock_backup_manager():
    """Mock backup manager"""
    return Mock()


@pytest.fixture
def recovery_config():
    """Create recovery configuration"""
    return RecoveryConfig(
        rto_minutes=5,
        rpo_minutes=1,
        max_retry_attempts=3,
        retry_delay_seconds=1,
        automatic_recovery_enabled=True,
        failure_detection_interval_seconds=5,
    )


@pytest.fixture
def orchestrator(mock_state_manager, mock_checkpoint_handler, mock_backup_manager, recovery_config):
    """Create RecoveryOrchestrator instance"""
    return RecoveryOrchestrator(
        mock_state_manager,
        mock_checkpoint_handler,
        mock_backup_manager,
        recovery_config,
    )


class TestRecoveryOrchestrator:
    """Test suite for RecoveryOrchestrator"""

    def test_initialization(self, orchestrator, recovery_config):
        """Test orchestrator initialization"""
        assert orchestrator.state_manager is not None
        assert orchestrator.checkpoint_handler is not None
        assert orchestrator.config == recovery_config
        assert orchestrator.detector is not None
        assert orchestrator.validator is not None
        assert orchestrator.alerter is not None
        assert len(orchestrator._failure_handlers) == 10

    def test_handler_registration(self, orchestrator):
        """Test failure handlers are registered"""
        expected_handlers = [
            FailureType.REDIS_DOWN,
            FailureType.POSTGRES_DOWN,
            FailureType.S3_UNAVAILABLE,
            FailureType.DATA_CORRUPTION,
            FailureType.TRADING_ENGINE_CRASH,
            FailureType.ML_MODEL_FAILURE,
            FailureType.MEMORY_OVERFLOW,
            FailureType.DISK_FULL,
            FailureType.NETWORK_PARTITION,
            FailureType.API_GATEWAY_DOWN,
        ]

        for failure_type in expected_handlers:
            assert failure_type in orchestrator._failure_handlers
            assert callable(orchestrator._failure_handlers[failure_type])

    @pytest.mark.asyncio
    async def test_detect_failures_delegates_to_detector(self, orchestrator):
        """Test failure detection delegates to FailureDetector"""
        orchestrator.detector.detect_failures = AsyncMock(return_value=[FailureType.REDIS_DOWN])

        failures = await orchestrator.detect_failures()

        assert failures == [FailureType.REDIS_DOWN]
        orchestrator.detector.detect_failures.assert_called_once()

    def test_is_critical_failure(self, orchestrator):
        """Test critical failure identification"""
        assert orchestrator._is_critical(FailureType.DATA_CORRUPTION)
        assert orchestrator._is_critical(FailureType.TRADING_ENGINE_CRASH)
        assert orchestrator._is_critical(FailureType.POSTGRES_DOWN)
        assert orchestrator._is_critical(FailureType.MEMORY_OVERFLOW)
        assert not orchestrator._is_critical(FailureType.S3_UNAVAILABLE)
        assert not orchestrator._is_critical(FailureType.REDIS_DOWN)

    def test_get_affected_components(self, orchestrator):
        """Test affected components mapping"""
        assert "cache" in orchestrator._get_affected_components(FailureType.REDIS_DOWN)
        assert "persistence" in orchestrator._get_affected_components(FailureType.POSTGRES_DOWN)
        assert "cold_storage" in orchestrator._get_affected_components(FailureType.S3_UNAVAILABLE)
        assert "order_execution" in orchestrator._get_affected_components(
            FailureType.TRADING_ENGINE_CRASH
        )

    def test_generate_operation_id(self, orchestrator):
        """Test operation ID generation"""
        op_id = orchestrator._generate_operation_id()

        assert op_id.startswith("REC_")
        assert len(op_id) > 15

    @pytest.mark.asyncio
    async def test_initiate_recovery_creates_operation(self, orchestrator):
        """Test recovery initiation creates operation"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis connection lost",
        )

        orchestrator._execute_recovery = AsyncMock(return_value=True)
        orchestrator.validator.validate_recovery = AsyncMock(return_value=True)
        orchestrator.alerter.send_alert = AsyncMock()

        operation = await orchestrator.initiate_recovery(failure_event, RecoveryMode.AUTOMATIC)

        assert operation.failure_event == failure_event
        assert operation.recovery_mode == RecoveryMode.AUTOMATIC
        assert operation.status == RecoveryStatus.COMPLETED
        assert operation.started_at is not None

    @pytest.mark.asyncio
    async def test_initiate_recovery_handles_run_errors(self, orchestrator):
        """Run-time failures should mark the operation as failed."""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="critical",
            affected_components=["cache"],
            error_message="Redis down",
        )

        orchestrator._run_recovery = AsyncMock(side_effect=RuntimeError("boom"))
        orchestrator.alerter.send_alert = AsyncMock()

        operation = await orchestrator.initiate_recovery(failure_event)

        assert operation.status == RecoveryStatus.FAILED
        assert operation.actions_taken[-1] == "Error: boom"

    @pytest.mark.asyncio
    async def test_initiate_recovery_already_in_progress(self, orchestrator):
        """Test recovery when already in progress"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        orchestrator._recovery_in_progress = True
        mock_operation = Mock()
        orchestrator._current_operation = mock_operation

        operation = await orchestrator.initiate_recovery(failure_event)

        assert operation == mock_operation

    @pytest.mark.asyncio
    async def test_execute_recovery_success(self, orchestrator):
        """Test successful recovery execution"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        mock_handler = AsyncMock(return_value=True)
        orchestrator._failure_handlers[FailureType.REDIS_DOWN] = mock_handler

        success = await orchestrator._execute_recovery(operation)

        assert success is True
        mock_handler.assert_called_once_with(operation)
        assert len(operation.actions_taken) > 0

    @pytest.mark.asyncio
    async def test_execute_recovery_with_retries(self, orchestrator):
        """Test recovery execution with retries"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        # Fail twice, succeed on third attempt
        mock_handler = AsyncMock(side_effect=[False, False, True])
        orchestrator._failure_handlers[FailureType.REDIS_DOWN] = mock_handler

        success = await orchestrator._execute_recovery(operation)

        assert success is True
        assert mock_handler.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_recovery_all_attempts_fail(self, orchestrator):
        """Test recovery execution when all attempts fail"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        mock_handler = AsyncMock(return_value=False)
        orchestrator._failure_handlers[FailureType.REDIS_DOWN] = mock_handler

        success = await orchestrator._execute_recovery(operation)

        assert success is False
        assert mock_handler.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_recovery_no_handler(self, orchestrator):
        """Test recovery execution with no registered handler"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        # Remove handler
        del orchestrator._failure_handlers[FailureType.REDIS_DOWN]

        success = await orchestrator._execute_recovery(operation)

        assert success is False

    @pytest.mark.asyncio
    async def test_execute_recovery_handler_raises(self, orchestrator):
        """Handler exceptions should be captured and logged."""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        orchestrator._failure_handlers[FailureType.REDIS_DOWN] = AsyncMock(
            side_effect=RuntimeError("handler exploded")
        )

        success = await orchestrator._execute_recovery(operation)

        assert success is False
        assert operation.actions_taken.count("Handler error: handler exploded") == orchestrator.config.max_retry_attempts

    @pytest.mark.asyncio
    async def test_complete_recovery_validates_successfully(self, orchestrator):
        """Test successful recovery completion with validation"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        orchestrator.validator.validate_recovery = AsyncMock(return_value=True)

        await orchestrator._complete_recovery(operation)

        assert operation.status == RecoveryStatus.COMPLETED
        assert operation.completed_at is not None
        assert operation.recovery_time_seconds is not None

    @pytest.mark.asyncio
    async def test_complete_recovery_validation_fails(self, orchestrator):
        """Test recovery completion when validation fails"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        orchestrator.validator.validate_recovery = AsyncMock(return_value=False)

        await orchestrator._complete_recovery(operation)

        assert operation.status == RecoveryStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_handle_recovery_failure_automatic_mode(self, orchestrator):
        """Test recovery failure handling in automatic mode"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        orchestrator.alerter.escalate_recovery = AsyncMock()

        await orchestrator._handle_recovery_failure(operation, RecoveryMode.AUTOMATIC)

        assert operation.status == RecoveryStatus.FAILED
        orchestrator.alerter.escalate_recovery.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_handle_recovery_failure_manual_mode(self, orchestrator):
        """Test recovery failure handling in manual mode"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.MANUAL)

        orchestrator.alerter.escalate_recovery = AsyncMock()

        await orchestrator._handle_recovery_failure(operation, RecoveryMode.MANUAL)

        assert operation.status == RecoveryStatus.FAILED
        orchestrator.alerter.escalate_recovery.assert_not_called()

    def test_cleanup_recovery_history(self, orchestrator):
        """Test recovery history cleanup"""
        # Add 150 operations
        for i in range(150):
            orchestrator._recovery_history.append(Mock())

        orchestrator._cleanup_recovery_history()

        assert len(orchestrator._recovery_history) == 100

    def test_get_recovery_stats_empty(self, orchestrator):
        """Test recovery stats with no history"""
        stats = orchestrator.get_recovery_stats()

        assert stats["total_recoveries"] == 0
        assert stats["success_rate"] == 0
        assert stats["average_recovery_time"] == 0

    def test_get_recovery_stats_with_history(self, orchestrator):
        """Test recovery stats with history"""
        # Add successful operations
        for i in range(3):
            op = Mock()
            op.status = RecoveryStatus.COMPLETED
            op.recovery_time_seconds = 10.0 * (i + 1)
            orchestrator._recovery_history.append(op)

        # Add failed operation
        failed_op = Mock()
        failed_op.status = RecoveryStatus.FAILED
        failed_op.recovery_time_seconds = None
        orchestrator._recovery_history.append(failed_op)

        stats = orchestrator.get_recovery_stats()

        assert stats["total_recoveries"] == 4
        assert stats["successful_recoveries"] == 3
        assert stats["success_rate"] == 75.0
        assert stats["average_recovery_time"] == 20.0  # (10+20+30)/3

    @pytest.mark.asyncio
    async def test_start_monitoring(self, orchestrator):
        """Test monitoring start"""
        orchestrator._monitoring_loop = AsyncMock()

        await orchestrator.start_monitoring()

        assert orchestrator._monitoring_task is not None

    @pytest.mark.asyncio
    async def test_start_monitoring_already_started(self, orchestrator):
        """Test starting monitoring when already started"""
        orchestrator._monitoring_task = Mock()

        await orchestrator.start_monitoring()

        # Should not create new task

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, orchestrator):
        """Test monitoring stop"""
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        orchestrator._monitoring_task = mock_task

        await orchestrator.stop_monitoring()

        mock_task.cancel.assert_called_once()
        assert orchestrator._monitoring_task is None

    @pytest.mark.asyncio
    async def test_stop_monitoring_handles_cancelled_error(self, orchestrator):
        """Stop should gracefully swallow CancelledError from task await."""

        class CancelOnce:
            def __init__(self) -> None:
                self.cancel_called = False

            def cancel(self) -> None:
                self.cancel_called = True

            def __await__(self):
                async def _raiser():
                    raise asyncio.CancelledError

                return _raiser().__await__()

        task = CancelOnce()
        orchestrator._monitoring_task = task

        await orchestrator.stop_monitoring()

        assert task.cancel_called is True
        assert orchestrator._monitoring_task is None

    @pytest.mark.asyncio
    async def test_monitoring_loop_detects_and_recovers(self, orchestrator):
        """Test monitoring loop detection and recovery"""
        orchestrator.detector.detect_failures = AsyncMock(
            side_effect=[
                [FailureType.DATA_CORRUPTION],
                [],  # No failures on second check
            ]
        )

        orchestrator._execute_recovery = AsyncMock(return_value=True)
        orchestrator.validator.validate_recovery = AsyncMock(return_value=True)
        orchestrator.alerter.send_alert = AsyncMock()

        # Run monitoring loop for short time
        task = asyncio.create_task(orchestrator._monitoring_loop())
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have attempted recovery
        orchestrator.detector.detect_failures.assert_called()

    @pytest.mark.asyncio
    async def test_monitoring_loop_handles_detector_errors(self, orchestrator, monkeypatch):
        """Detector exceptions should trigger backoff sleep."""
        orchestrator.detector.detect_failures = AsyncMock(side_effect=RuntimeError("boom"))

        sleep_calls: list[int] = []

        async def fake_sleep(duration: int) -> None:
            sleep_calls.append(duration)
            raise asyncio.CancelledError

        monkeypatch.setattr("bot_v2.state.recovery.orchestrator.asyncio.sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await orchestrator._monitoring_loop()

        assert sleep_calls == [orchestrator.config.failure_detection_interval_seconds * 2]

    @pytest.mark.asyncio
    async def test_rto_compliance_warning(self, orchestrator):
        """Test RTO compliance warning"""
        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis down",
        )

        operation = orchestrator._create_recovery_operation(failure_event, RecoveryMode.AUTOMATIC)

        # Simulate long recovery time
        operation.recovery_time_seconds = 400  # > 5 minutes

        orchestrator.validator.validate_recovery = AsyncMock(return_value=True)

        with patch("bot_v2.state.recovery.orchestrator.logger") as mock_logger:
            await orchestrator._complete_recovery(operation)

            # Should log warning about RTO
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("exceeded RTO" in str(call) for call in warning_calls)


@pytest.mark.asyncio
async def test_detect_and_recover_initiates_when_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubOrchestrator:
        failures: list[FailureType] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.failures = list(self.__class__.failures)
            self.last_event: FailureEvent | None = None

        async def detect_failures(self) -> list[FailureType]:
            return self.failures

        def _is_critical(self, failure: FailureType) -> bool:
            return True

        def _get_affected_components(self, failure: FailureType) -> list[str]:
            return ["system"]

        async def initiate_recovery(self, event: FailureEvent, mode: RecoveryMode = RecoveryMode.AUTOMATIC):
            self.last_event = event
            return "operation"

    StubOrchestrator.failures = [FailureType.POSTGRES_DOWN]
    monkeypatch.setattr(
        "bot_v2.state.recovery.orchestrator.RecoveryOrchestrator", StubOrchestrator
    )

    result = await detect_and_recover(SimpleNamespace(), SimpleNamespace())

    assert result == "operation"


@pytest.mark.asyncio
async def test_detect_and_recover_returns_none_for_non_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubOrchestrator:
        failures: list[FailureType] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.failures = list(self.__class__.failures)

        async def detect_failures(self) -> list[FailureType]:
            return self.failures

        def _is_critical(self, failure: FailureType) -> bool:
            return False

        def _get_affected_components(self, failure: FailureType) -> list[str]:
            return ["system"]

        async def initiate_recovery(self, event: FailureEvent, mode: RecoveryMode = RecoveryMode.AUTOMATIC):
            raise AssertionError("Should not be called")

    StubOrchestrator.failures = [FailureType.S3_UNAVAILABLE]
    monkeypatch.setattr(
        "bot_v2.state.recovery.orchestrator.RecoveryOrchestrator", StubOrchestrator
    )

    result = await detect_and_recover(SimpleNamespace(), SimpleNamespace())

    assert result is None
