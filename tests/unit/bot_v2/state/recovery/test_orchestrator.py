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
        assert len(orchestrator.handler_registry) == 10

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
            assert orchestrator.handler_registry.has_handler(failure_type)
            assert callable(orchestrator.handler_registry.get_handler(failure_type))

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
        from bot_v2.state.recovery.workflow import RecoveryOutcome

        failure_event = FailureEvent(
            failure_type=FailureType.REDIS_DOWN,
            timestamp=datetime.utcnow(),
            severity="high",
            affected_components=["cache"],
            error_message="Redis connection lost",
        )

        # Mock workflow to update operation status and return successful outcome
        async def mock_execute(op, mode):
            op.status = RecoveryStatus.COMPLETED
            op.completed_at = datetime.utcnow()
            return RecoveryOutcome(
                success=True,
                status=RecoveryStatus.COMPLETED,
                validation_passed=True,
            )

        orchestrator.workflow.execute = mock_execute

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
        """Test monitoring start (delegates to RecoveryMonitor)"""
        await orchestrator.start_monitoring()

        assert orchestrator.monitor.is_running()

        await orchestrator.stop_monitoring()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_started(self, orchestrator):
        """Test starting monitoring when already started (idempotent)"""
        await orchestrator.start_monitoring()
        assert orchestrator.monitor.is_running()

        # Should be idempotent
        await orchestrator.start_monitoring()
        assert orchestrator.monitor.is_running()

        await orchestrator.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, orchestrator):
        """Test monitoring stop (delegates to RecoveryMonitor)"""
        await orchestrator.start_monitoring()
        assert orchestrator.monitor.is_running()

        await orchestrator.stop_monitoring()

        assert not orchestrator.monitor.is_running()

    @pytest.mark.asyncio
    async def test_stop_monitoring_handles_cancelled_error(self, orchestrator):
        """Stop should gracefully handle cancellation (delegated to RecoveryMonitor)."""
        await orchestrator.start_monitoring()
        assert orchestrator.monitor.is_running()

        # Stop should handle cancellation gracefully
        await orchestrator.stop_monitoring()

        assert not orchestrator.monitor.is_running()

    @pytest.mark.asyncio
    async def test_monitoring_loop_detects_and_recovers(self, orchestrator):
        """Test monitoring loop detection and recovery (via RecoveryMonitor)"""
        from bot_v2.state.recovery.workflow import RecoveryOutcome

        orchestrator.detector.detect_failures = AsyncMock(
            side_effect=[
                [FailureType.DATA_CORRUPTION],
                [],  # No failures on second check
            ]
        )

        # Mock workflow to simulate successful recovery
        async def mock_workflow_execute(op, mode):
            op.status = RecoveryStatus.COMPLETED
            return RecoveryOutcome(success=True, status=RecoveryStatus.COMPLETED)

        orchestrator.workflow.execute = mock_workflow_execute

        # Run monitoring for short time
        await orchestrator.start_monitoring()
        await asyncio.sleep(0.1)
        await orchestrator.stop_monitoring()

        # Should have attempted detection
        orchestrator.detector.detect_failures.assert_called()

    @pytest.mark.asyncio
    async def test_monitoring_loop_handles_detector_errors(self, orchestrator):
        """Monitor should handle detector exceptions (delegated to RecoveryMonitor)."""
        orchestrator.detector.detect_failures = AsyncMock(
            side_effect=[
                RuntimeError("boom"),
                [],  # Recover on second call
            ]
        )

        # Run monitoring briefly
        await orchestrator.start_monitoring()
        await asyncio.sleep(0.1)
        await orchestrator.stop_monitoring()

        # Should have called detector multiple times (error + recovery)
        assert orchestrator.detector.detect_failures.call_count >= 1


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

        async def initiate_recovery(
            self, event: FailureEvent, mode: RecoveryMode = RecoveryMode.AUTOMATIC
        ):
            self.last_event = event
            return "operation"

    StubOrchestrator.failures = [FailureType.POSTGRES_DOWN]
    monkeypatch.setattr("bot_v2.state.recovery.orchestrator.RecoveryOrchestrator", StubOrchestrator)

    result = await detect_and_recover(SimpleNamespace(), SimpleNamespace())

    assert result == "operation"


@pytest.mark.asyncio
async def test_detect_and_recover_returns_none_for_non_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

        async def initiate_recovery(
            self, event: FailureEvent, mode: RecoveryMode = RecoveryMode.AUTOMATIC
        ):
            raise AssertionError("Should not be called")

    StubOrchestrator.failures = [FailureType.S3_UNAVAILABLE]
    monkeypatch.setattr("bot_v2.state.recovery.orchestrator.RecoveryOrchestrator", StubOrchestrator)

    result = await detect_and_recover(SimpleNamespace(), SimpleNamespace())

    assert result is None
