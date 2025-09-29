"""
Workflow Execution Tracker for Bot V2 Monitoring System

Tracks workflow execution lifecycle, records step-by-step progress,
measures execution times, stores workflow history, and generates execution reports.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class WorkflowStatus(Enum):
    """Workflow execution status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowExecution:
    """Data class representing a workflow execution instance"""

    id: str
    workflow_name: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: datetime | None = None
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class WorkflowTracker:
    """
    Tracks workflow execution lifecycle and provides monitoring capabilities
    """

    def __init__(self) -> None:
        self.executions: dict[str, WorkflowExecution] = {}
        self.execution_history: list[WorkflowExecution] = []
        self.step_timings: dict[str, dict[str, dict[str, Any]]] = {}
        self.workflow_registry: dict[str, dict[str, Any]] = {}

    def register_workflow(
        self, workflow_name: str, expected_steps: int, description: str = ""
    ) -> None:
        """Register a workflow type with expected step count"""
        self.workflow_registry[workflow_name] = {
            "expected_steps": expected_steps,
            "description": description,
            "registered_at": datetime.now(),
        }

    def start_workflow(self, workflow_name: str, context: dict[str, Any] | None = None) -> str:
        """Start tracking a new workflow execution"""
        execution_id = str(uuid.uuid4())

        # Get expected steps from registry
        expected_steps = 0
        if workflow_name in self.workflow_registry:
            expected_steps = self.workflow_registry[workflow_name]["expected_steps"]

        execution = WorkflowExecution(
            id=execution_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
            steps_total=expected_steps,
            context=context or {},
        )

        self.executions[execution_id] = execution
        return execution_id

    def track_step(
        self,
        execution_id: str,
        step_name: str,
        status: str,
        duration: float = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track completion of a workflow step"""
        if execution_id not in self.executions:
            return

        execution = self.executions[execution_id]

        # Update step counts
        if status == "completed":
            execution.steps_completed += 1
        elif status == "failed":
            execution.steps_failed += 1

        # Record step timing and metadata
        if execution_id not in self.step_timings:
            self.step_timings[execution_id] = {}

        self.step_timings[execution_id][step_name] = {
            "duration": duration,
            "status": status,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }

        # Update execution metrics
        total_duration = sum(step["duration"] for step in self.step_timings[execution_id].values())
        execution.metrics["current_duration"] = total_duration

    def add_error(self, execution_id: str, error_message: str) -> None:
        """Add an error message to the workflow execution"""
        if execution_id in self.executions:
            self.executions[execution_id].errors.append(error_message)

    def update_context(self, execution_id: str, context_updates: dict[str, Any]) -> None:
        """Update the execution context with new data"""
        if execution_id in self.executions:
            self.executions[execution_id].context.update(context_updates)

    def complete_workflow(
        self,
        execution_id: str,
        status: WorkflowStatus,
        final_metrics: dict[str, float] | None = None,
    ) -> None:
        """Mark a workflow as completed and move to history"""
        if execution_id not in self.executions:
            return

        execution = self.executions[execution_id]
        execution.status = status
        execution.completed_at = datetime.now()

        # Calculate total duration
        duration = (execution.completed_at - execution.started_at).total_seconds()
        execution.metrics["total_duration"] = duration

        # Add any final metrics
        if final_metrics:
            execution.metrics.update(final_metrics)

        # Move to history and clean up active executions
        self.execution_history.append(execution)
        del self.executions[execution_id]

        # Maintain history size limit
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)

        # Clean up old step timings
        if execution_id in self.step_timings:
            del self.step_timings[execution_id]

    def get_execution_status(self, execution_id: str) -> dict[str, Any] | None:
        """Get current status of an active execution"""
        if execution_id not in self.executions:
            return None

        execution = self.executions[execution_id]
        progress = 0.0
        if execution.steps_total > 0:
            progress = execution.steps_completed / execution.steps_total

        return {
            "id": execution.id,
            "workflow": execution.workflow_name,
            "status": execution.status.value,
            "progress": progress,
            "steps_completed": execution.steps_completed,
            "steps_total": execution.steps_total,
            "steps_failed": execution.steps_failed,
            "current_duration": execution.metrics.get("current_duration", 0),
            "error_count": len(execution.errors),
        }

    def get_execution_report(self, execution_id: str) -> dict[str, Any]:
        """Generate comprehensive execution report"""
        # Check active executions first
        execution = None
        if execution_id in self.executions:
            execution = self.executions[execution_id]
        else:
            # Check history
            execution = next((e for e in self.execution_history if e.id == execution_id), None)

        if not execution:
            return {}

        progress = 0.0
        if execution.steps_total > 0:
            progress = execution.steps_completed / execution.steps_total

        return {
            "id": execution.id,
            "workflow": execution.workflow_name,
            "status": execution.status.value,
            "progress": progress,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration": execution.metrics.get(
                "total_duration", execution.metrics.get("current_duration", 0)
            ),
            "steps": {
                "total": execution.steps_total,
                "completed": execution.steps_completed,
                "failed": execution.steps_failed,
            },
            "context": execution.context,
            "errors": execution.errors,
            "metrics": execution.metrics,
            "step_timings": self.step_timings.get(execution_id, {}),
        }

    def get_workflow_stats(self, workflow_name: str) -> dict[str, Any]:
        """Calculate statistics for a specific workflow type"""
        relevant_executions = [
            e for e in self.execution_history if e.workflow_name == workflow_name
        ]

        if not relevant_executions:
            return {
                "workflow_name": workflow_name,
                "total_executions": 0,
                "success_rate": 0,
                "failure_rate": 0,
                "avg_duration": 0,
                "last_execution": None,
            }

        success_count = sum(1 for e in relevant_executions if e.status == WorkflowStatus.COMPLETED)
        failure_count = sum(1 for e in relevant_executions if e.status == WorkflowStatus.FAILED)
        total_executions = len(relevant_executions)

        avg_duration = (
            sum(e.metrics.get("total_duration", 0) for e in relevant_executions) / total_executions
        )

        return {
            "workflow_name": workflow_name,
            "total_executions": total_executions,
            "success_rate": success_count / total_executions,
            "failure_rate": failure_count / total_executions,
            "avg_duration": avg_duration,
            "last_execution": relevant_executions[-1].started_at.isoformat(),
            "registry_info": self.workflow_registry.get(workflow_name, {}),
        }

    def get_active_workflows(self) -> list[dict[str, Any]]:
        """Get list of currently active workflow executions"""
        active: list[dict[str, Any]] = []
        for execution_id in self.executions:
            status = self.get_execution_status(execution_id)
            if status is not None:
                active.append(status)
        return active

    def get_all_workflow_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered workflows"""
        stats = {}

        # Get stats for registered workflows
        for workflow_name in self.workflow_registry.keys():
            stats[workflow_name] = self.get_workflow_stats(workflow_name)

        # Include unregistered workflows from history
        for execution in self.execution_history:
            if execution.workflow_name not in stats:
                stats[execution.workflow_name] = self.get_workflow_stats(execution.workflow_name)

        return stats

    def cleanup_old_data(self, max_age_days: int = 7) -> None:
        """Clean up old execution data beyond specified age"""
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

        self.execution_history = [
            e for e in self.execution_history if e.started_at.timestamp() > cutoff_date
        ]
