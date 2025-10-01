"""Tests for workflow tracker"""

import pytest
from datetime import datetime
from bot_v2.monitoring.workflow_tracker import (
    WorkflowTracker,
    WorkflowStatus,
    WorkflowExecution,
)


class TestWorkflowTracker:
    """Test suite for WorkflowTracker"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = WorkflowTracker()

        assert tracker.executions == {}
        assert tracker.execution_history == []
        assert tracker.workflow_registry == {}

    def test_register_workflow(self):
        """Test registering a workflow"""
        tracker = WorkflowTracker()

        tracker.register_workflow("data_pipeline", 5, "ETL pipeline")

        assert "data_pipeline" in tracker.workflow_registry
        assert tracker.workflow_registry["data_pipeline"]["expected_steps"] == 5
        assert tracker.workflow_registry["data_pipeline"]["description"] == "ETL pipeline"

    def test_start_workflow(self):
        """Test starting a workflow"""
        tracker = WorkflowTracker()
        tracker.register_workflow("test_workflow", 3)

        execution_id = tracker.start_workflow("test_workflow")

        assert execution_id in tracker.executions
        execution = tracker.executions[execution_id]
        assert execution.workflow_name == "test_workflow"
        assert execution.status == WorkflowStatus.RUNNING
        assert execution.steps_total == 3

    def test_start_workflow_with_context(self):
        """Test starting workflow with context"""
        tracker = WorkflowTracker()

        context = {"user_id": "123", "symbol": "BTC-USD"}
        execution_id = tracker.start_workflow("trade_execution", context=context)

        execution = tracker.executions[execution_id]
        assert execution.context == context

    def test_track_step_completed(self):
        """Test tracking a completed step"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.track_step(execution_id, "step1", "completed", duration=1.5)

        execution = tracker.executions[execution_id]
        assert execution.steps_completed == 1
        assert execution.steps_failed == 0

    def test_track_step_failed(self):
        """Test tracking a failed step"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.track_step(execution_id, "step1", "failed", duration=0.5)

        execution = tracker.executions[execution_id]
        assert execution.steps_completed == 0
        assert execution.steps_failed == 1

    def test_track_step_with_metadata(self):
        """Test tracking step with metadata"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        metadata = {"records_processed": 1000, "errors": 0}
        tracker.track_step(
            execution_id, "process_data", "completed", duration=2.0, metadata=metadata
        )

        step_info = tracker.step_timings[execution_id]["process_data"]
        assert step_info["metadata"] == metadata
        assert step_info["duration"] == 2.0

    def test_track_step_updates_duration_metric(self):
        """Test that tracking steps updates duration metric"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.track_step(execution_id, "step1", "completed", duration=1.0)
        tracker.track_step(execution_id, "step2", "completed", duration=2.0)

        execution = tracker.executions[execution_id]
        assert execution.metrics["current_duration"] == 3.0

    def test_track_step_invalid_execution(self):
        """Test tracking step for invalid execution ID"""
        tracker = WorkflowTracker()

        # Should not raise, just return
        tracker.track_step("invalid_id", "step1", "completed")

    def test_add_error(self):
        """Test adding an error to execution"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.add_error(execution_id, "Connection failed")
        tracker.add_error(execution_id, "Timeout exceeded")

        execution = tracker.executions[execution_id]
        assert len(execution.errors) == 2
        assert "Connection failed" in execution.errors

    def test_update_context(self):
        """Test updating execution context"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow", context={"a": 1})

        tracker.update_context(execution_id, {"b": 2, "c": 3})

        execution = tracker.executions[execution_id]
        assert execution.context == {"a": 1, "b": 2, "c": 3}

    def test_complete_workflow_success(self):
        """Test completing workflow successfully"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")
        tracker.track_step(execution_id, "step1", "completed", duration=1.0)

        tracker.complete_workflow(execution_id, WorkflowStatus.COMPLETED)

        # Execution should be moved to history
        assert execution_id not in tracker.executions
        assert len(tracker.execution_history) == 1
        execution = tracker.execution_history[0]
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.completed_at is not None
        assert "total_duration" in execution.metrics

    def test_complete_workflow_with_final_metrics(self):
        """Test completing workflow with final metrics"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        final_metrics = {"total_records": 5000, "success_rate": 0.99}
        tracker.complete_workflow(
            execution_id, WorkflowStatus.COMPLETED, final_metrics=final_metrics
        )

        execution = tracker.execution_history[0]
        assert execution.metrics["total_records"] == 5000
        assert execution.metrics["success_rate"] == 0.99

    def test_complete_workflow_failed(self):
        """Test completing workflow with failure status"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")
        tracker.add_error(execution_id, "Critical error occurred")

        tracker.complete_workflow(execution_id, WorkflowStatus.FAILED)

        execution = tracker.execution_history[0]
        assert execution.status == WorkflowStatus.FAILED
        assert len(execution.errors) == 1

    def test_get_execution_status(self):
        """Test getting execution status by ID"""
        tracker = WorkflowTracker()
        tracker.register_workflow("test_workflow", 3)
        execution_id = tracker.start_workflow("test_workflow")

        status = tracker.get_execution_status(execution_id)

        assert status is not None
        assert status["id"] == execution_id
        assert status["workflow"] == "test_workflow"
        assert status["status"] == "running"

    def test_get_workflow_stats(self):
        """Test getting workflow statistics"""
        tracker = WorkflowTracker()
        tracker.register_workflow("test_workflow", 3)

        # Run multiple executions
        for i in range(5):
            exec_id = tracker.start_workflow("test_workflow")
            tracker.track_step(exec_id, "step1", "completed", duration=1.0)
            status = WorkflowStatus.COMPLETED if i < 4 else WorkflowStatus.FAILED
            tracker.complete_workflow(exec_id, status)

        stats = tracker.get_workflow_stats("test_workflow")

        assert stats["total_executions"] == 5
        assert stats["success_rate"] == 0.8  # 4 out of 5
        assert stats["failure_rate"] == 0.2  # 1 out of 5
        assert "avg_duration" in stats

    def test_get_active_workflows(self):
        """Test getting active workflows"""
        tracker = WorkflowTracker()

        exec_id1 = tracker.start_workflow("workflow1")
        exec_id2 = tracker.start_workflow("workflow2")
        exec_id3 = tracker.start_workflow("workflow3")

        # Complete one
        tracker.complete_workflow(exec_id3, WorkflowStatus.COMPLETED)

        active = tracker.get_active_workflows()

        assert len(active) == 2
        # active is a list of dicts, not WorkflowExecution objects
        assert exec_id1 in [e["id"] for e in active]
        assert exec_id2 in [e["id"] for e in active]

    def test_history_limit(self):
        """Test execution history limit"""
        tracker = WorkflowTracker()

        # Run many executions
        for i in range(150):
            exec_id = tracker.start_workflow("test_workflow")
            tracker.complete_workflow(exec_id, WorkflowStatus.COMPLETED)

        # Should maintain reasonable history size
        assert len(tracker.execution_history) <= 100

    def test_get_step_timings(self):
        """Test getting step timings for execution"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.track_step(execution_id, "step1", "completed", duration=1.5)
        tracker.track_step(execution_id, "step2", "completed", duration=2.5)

        # Access step_timings dict directly
        timings = tracker.step_timings[execution_id]

        assert "step1" in timings
        assert "step2" in timings
        assert timings["step1"]["duration"] == 1.5

    def test_workflow_execution_dataclass(self):
        """Test WorkflowExecution dataclass"""
        execution = WorkflowExecution(
            id="test-123",
            workflow_name="test_workflow",
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
        )

        assert execution.id == "test-123"
        assert execution.status == WorkflowStatus.RUNNING
        assert execution.steps_total == 0
        assert execution.context == {}
        assert execution.errors == []

    def test_workflow_status_enum(self):
        """Test WorkflowStatus enum values"""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    def test_cancel_workflow(self):
        """Test cancelling a workflow"""
        tracker = WorkflowTracker()
        execution_id = tracker.start_workflow("test_workflow")

        tracker.complete_workflow(execution_id, WorkflowStatus.CANCELLED)

        # Should be moved to history
        execution = tracker.execution_history[0]
        assert execution.status == WorkflowStatus.CANCELLED
