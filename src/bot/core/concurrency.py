"""
GPT-Trader Unified Concurrency Framework

Centralized concurrency management providing:
- Thread pool management with lifecycle coordination
- Async/await patterns for I/O-bound operations
- Background task scheduling and execution
- Inter-component communication via message queues
- Graceful shutdown handling across all threads
- Performance monitoring and resource management

This eliminates inconsistent threading patterns throughout the codebase
and ensures proper resource cleanup and coordination between components.
"""

import logging
import queue
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import Empty, PriorityQueue, Queue
from typing import Any, Generic, TypeVar

from .base import BaseComponent
from .config import get_config
from .exceptions import ComponentException, raise_validation_error

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ThreadPoolType(Enum):
    """Thread pool types for different workload patterns"""

    IO_BOUND = "io_bound"  # I/O operations, API calls, file operations
    CPU_BOUND = "cpu_bound"  # Calculations, data processing
    MONITORING = "monitoring"  # Health checks, metrics collection
    BACKGROUND = "background"  # Cleanup, maintenance tasks


class TaskPriority(Enum):
    """Task execution priority levels"""

    CRITICAL = 1  # System critical operations
    HIGH = 2  # Trading operations
    NORMAL = 3  # Regular operations
    LOW = 4  # Background tasks


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskMetrics:
    """Task execution metrics"""

    task_id: str
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: str | None = None

    @property
    def execution_time(self) -> timedelta | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ScheduledTask:
    """Scheduled task definition"""

    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    run_at: datetime | None = None
    repeat_interval: timedelta | None = None
    max_retries: int = 0
    retry_count: int = 0
    component_id: str | None = None

    def __lt__(self, other):
        """Priority queue ordering"""
        if not isinstance(other, ScheduledTask):
            return NotImplemented
        return self.priority.value < other.priority.value


class AsyncTaskResult(Generic[T]):
    """Result container for async task execution"""

    def __init__(self, future: Future[T]) -> None:
        self._future = future
        self._task_id: str | None = None

    def result(self, timeout: float | None = None) -> T:
        """Get task result with optional timeout"""
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> Exception | None:
        """Get task exception with optional timeout"""
        return self._future.exception(timeout=timeout)

    def done(self) -> bool:
        """Check if task is completed"""
        return self._future.done()

    def cancelled(self) -> bool:
        """Check if task was cancelled"""
        return self._future.cancelled()

    def cancel(self) -> bool:
        """Attempt to cancel task"""
        return self._future.cancel()

    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        """Add callback for task completion"""
        self._future.add_done_callback(callback)


class IMessageHandler(ABC):
    """Interface for message handlers"""

    @abstractmethod
    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle incoming message and optionally return response"""
        pass


@dataclass
class ThreadPoolConfig:
    """Configuration for thread pools"""

    pool_type: ThreadPoolType
    max_workers: int
    thread_name_prefix: str
    keep_alive_seconds: float = 60.0
    queue_maxsize: int = 1000


class ManagedThreadPool:
    """Managed thread pool with monitoring and lifecycle management"""

    def __init__(self, config: ThreadPoolConfig, concurrency_manager: "ConcurrencyManager") -> None:
        self.config = config
        self.concurrency_manager = concurrency_manager

        # Thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix=config.thread_name_prefix
        )

        # Task tracking
        self.active_tasks: dict[str, Future] = {}
        self.completed_tasks: list[TaskMetrics] = []
        self.task_counter = 0
        self.pool_lock = threading.RLock()

        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0

        logger.info(
            f"Created {config.pool_type.value} thread pool with {config.max_workers} workers"
        )

    def submit_task(
        self,
        function: Callable[..., T],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: str | None = None,
        component_id: str | None = None,
        **kwargs,
    ) -> AsyncTaskResult[T]:
        """Submit task for execution"""

        with self.pool_lock:
            # Generate task ID
            if not task_id:
                self.task_counter += 1
                task_id = f"{self.config.pool_type.value}_task_{self.task_counter}"

            # Create task metrics
            metrics = TaskMetrics(task_id=task_id, submitted_at=datetime.now())

            # Wrap function for monitoring
            def monitored_function():
                metrics.started_at = datetime.now()
                metrics.status = TaskStatus.RUNNING

                try:
                    result = function(*args, **kwargs)
                    metrics.status = TaskStatus.COMPLETED
                    metrics.completed_at = datetime.now()

                    if metrics.started_at:
                        duration = (
                            metrics.completed_at - metrics.started_at
                        ).total_seconds() * 1000
                        metrics.duration_ms = duration
                        self.total_execution_time += duration

                    with self.pool_lock:
                        self.tasks_completed += 1
                        self.completed_tasks.append(metrics)
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                    return result

                except Exception as e:
                    metrics.status = TaskStatus.FAILED
                    metrics.completed_at = datetime.now()
                    metrics.error_message = str(e)

                    with self.pool_lock:
                        self.tasks_failed += 1
                        self.completed_tasks.append(metrics)
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                    logger.error(f"Task {task_id} failed: {str(e)}")
                    raise

            # Submit to executor
            future = self.executor.submit(monitored_function)
            self.active_tasks[task_id] = future
            self.tasks_submitted += 1

            # Create result wrapper
            result = AsyncTaskResult(future)
            result._task_id = task_id

            return result

    def get_pool_stats(self) -> dict[str, Any]:
        """Get thread pool statistics"""
        with self.pool_lock:
            avg_execution_time = self.total_execution_time / max(1, self.tasks_completed)

            return {
                "pool_type": self.config.pool_type.value,
                "max_workers": self.config.max_workers,
                "active_tasks": len(self.active_tasks),
                "tasks_submitted": self.tasks_submitted,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "success_rate": (self.tasks_completed / max(1, self.tasks_submitted) * 100),
                "avg_execution_time_ms": avg_execution_time,
                "queue_size": getattr(self.executor._work_queue, "qsize", lambda: 0)(),
            }

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown thread pool gracefully"""
        logger.info(f"Shutting down {self.config.pool_type.value} thread pool...")

        # Cancel pending tasks
        with self.pool_lock:
            for _task_id, future in self.active_tasks.items():
                if not future.done():
                    future.cancel()

        # Shutdown executor
        self.executor.shutdown(wait=wait, timeout=timeout)

        logger.info(f"Thread pool {self.config.pool_type.value} shut down")


class TaskScheduler:
    """Background task scheduler with priority queues"""

    def __init__(self, concurrency_manager: "ConcurrencyManager") -> None:
        self.concurrency_manager = concurrency_manager

        # Task queues
        self.pending_tasks = PriorityQueue()
        self.scheduled_tasks: dict[str, ScheduledTask] = {}
        self.recurring_tasks: dict[str, ScheduledTask] = {}

        # Scheduler state
        self.is_running = False
        self.scheduler_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

        logger.info("Task scheduler initialized")

    def schedule_task(
        self,
        task_id: str,
        function: Callable,
        run_at: datetime | None = None,
        delay: timedelta | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 0,
        component_id: str | None = None,
        *args,
        **kwargs,
    ) -> str:
        """Schedule a task for execution"""

        if run_at is None:
            if delay:
                run_at = datetime.now() + delay
            else:
                run_at = datetime.now()

        task = ScheduledTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            run_at=run_at,
            max_retries=max_retries,
            component_id=component_id,
        )

        self.scheduled_tasks[task_id] = task
        self.pending_tasks.put((run_at.timestamp(), task))

        logger.debug(f"Scheduled task {task_id} for {run_at}")
        return task_id

    def schedule_recurring_task(
        self,
        task_id: str,
        function: Callable,
        interval: timedelta,
        priority: TaskPriority = TaskPriority.NORMAL,
        component_id: str | None = None,
        *args,
        **kwargs,
    ) -> str:
        """Schedule a recurring task"""

        task = ScheduledTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            run_at=datetime.now() + interval,
            repeat_interval=interval,
            component_id=component_id,
        )

        self.recurring_tasks[task_id] = task
        self.scheduled_tasks[task_id] = task
        self.pending_tasks.put((task.run_at.timestamp(), task))

        logger.info(f"Scheduled recurring task {task_id} every {interval}")
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            if task_id in self.recurring_tasks:
                del self.recurring_tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        return False

    def start(self) -> None:
        """Start the task scheduler"""
        if self.is_running:
            return

        self.is_running = True
        self.shutdown_event.clear()

        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="task_scheduler", daemon=True
        )
        self.scheduler_thread.start()

        logger.info("Task scheduler started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the task scheduler"""
        if not self.is_running:
            return

        logger.info("Stopping task scheduler...")
        self.is_running = False
        self.shutdown_event.set()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=timeout)

        logger.info("Task scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        logger.info("Task scheduler loop started")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()

                # Process pending tasks
                ready_tasks = []

                # Check for ready tasks (non-blocking)
                while True:
                    try:
                        timestamp, task = self.pending_tasks.get_nowait()

                        if timestamp <= current_time.timestamp():
                            ready_tasks.append(task)
                        else:
                            # Put back if not ready
                            self.pending_tasks.put((timestamp, task))
                            break
                    except Empty:
                        break

                # Execute ready tasks
                for task in ready_tasks:
                    try:
                        self._execute_scheduled_task(task)
                    except Exception as e:
                        logger.error(f"Error executing scheduled task {task.task_id}: {str(e)}")

                # Sleep briefly before next check
                if not self.shutdown_event.wait(1.0):
                    continue
                else:
                    break

            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                time.sleep(1.0)

        logger.info("Task scheduler loop stopped")

    def _execute_scheduled_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task"""
        try:
            # Submit to appropriate thread pool
            pool_type = ThreadPoolType.BACKGROUND
            if task.component_id:
                # Determine pool based on component type
                if "monitor" in task.component_id.lower():
                    pool_type = ThreadPoolType.MONITORING
                elif "trading" in task.component_id.lower():
                    pool_type = ThreadPoolType.IO_BOUND

            self.concurrency_manager.submit_task(
                pool_type=pool_type,
                function=task.function,
                task_id=task.task_id,
                component_id=task.component_id,
                priority=task.priority,
                *task.args,
                **task.kwargs,
            )

            logger.debug(f"Executed scheduled task {task.task_id}")

            # Handle recurring tasks
            if task.repeat_interval and task.task_id in self.recurring_tasks:
                next_run = datetime.now() + task.repeat_interval
                task.run_at = next_run
                self.pending_tasks.put((next_run.timestamp(), task))

        except Exception as e:
            logger.error(f"Failed to execute scheduled task {task.task_id}: {str(e)}")

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                retry_delay = timedelta(seconds=2**task.retry_count)  # Exponential backoff
                task.run_at = datetime.now() + retry_delay
                self.pending_tasks.put((task.run_at.timestamp(), task))
                logger.info(
                    f"Retrying task {task.task_id} in {retry_delay} (attempt {task.retry_count + 1})"
                )


class MessageQueue:
    """Thread-safe message queue for inter-component communication"""

    def __init__(self, queue_id: str, maxsize: int = 1000) -> None:
        self.queue_id = queue_id
        self.queue = Queue(maxsize=maxsize)
        self.subscribers: dict[str, IMessageHandler] = {}
        self.message_count = 0
        self.queue_lock = threading.RLock()

        logger.debug(f"Created message queue: {queue_id}")

    def subscribe(self, subscriber_id: str, handler: IMessageHandler) -> None:
        """Subscribe to queue messages"""
        with self.queue_lock:
            self.subscribers[subscriber_id] = handler
        logger.debug(f"Subscriber {subscriber_id} subscribed to queue {self.queue_id}")

    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from queue messages"""
        with self.queue_lock:
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
        logger.debug(f"Subscriber {subscriber_id} unsubscribed from queue {self.queue_id}")

    def publish(self, message: dict[str, Any], timeout: float | None = None) -> bool:
        """Publish message to queue"""
        try:
            self.queue.put(message, timeout=timeout)
            with self.queue_lock:
                self.message_count += 1
            return True
        except queue.Full:
            # Queue is full - message could not be published
            logger.debug(f"Queue {self.queue_id} is full, message not published")
            return False
        except Exception as e:
            # Unexpected error during message publishing
            logger.warning(f"Failed to publish message to queue {self.queue_id}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics"""
        with self.queue_lock:
            return {
                "queue_id": self.queue_id,
                "queue_size": self.queue.qsize(),
                "subscriber_count": len(self.subscribers),
                "total_messages": self.message_count,
            }


class ConcurrencyManager:
    """
    Unified concurrency management for GPT-Trader

    Provides centralized thread pool management, task scheduling,
    and inter-component messaging for all system components.
    """

    def __init__(self) -> None:
        """Initialize concurrency manager"""

        # Configuration
        self.config = get_config()

        # Thread pools
        self.thread_pools: dict[ThreadPoolType, ManagedThreadPool] = {}
        self.pool_configs = self._create_pool_configs()

        # Task management
        self.task_scheduler = TaskScheduler(self)
        self.message_queues: dict[str, MessageQueue] = {}

        # Lifecycle management
        self.is_initialized = False
        self.shutdown_event = threading.Event()
        self.manager_lock = threading.RLock()

        # Monitoring
        self.start_time = datetime.now()
        self.component_registry: dict[str, weakref.ref] = {}

        self._initialize_thread_pools()
        logger.info("Concurrency manager initialized")

    def _create_pool_configs(self) -> dict[ThreadPoolType, ThreadPoolConfig]:
        """Create thread pool configurations"""

        # Calculate pool sizes based on system resources
        import os

        cpu_count = os.cpu_count() or 4

        return {
            ThreadPoolType.IO_BOUND: ThreadPoolConfig(
                pool_type=ThreadPoolType.IO_BOUND,
                max_workers=min(32, (cpu_count + 4) * 2),  # I/O can handle more workers
                thread_name_prefix="gpt-trader-io",
            ),
            ThreadPoolType.CPU_BOUND: ThreadPoolConfig(
                pool_type=ThreadPoolType.CPU_BOUND,
                max_workers=cpu_count,  # CPU-bound should match CPU cores
                thread_name_prefix="gpt-trader-cpu",
            ),
            ThreadPoolType.MONITORING: ThreadPoolConfig(
                pool_type=ThreadPoolType.MONITORING,
                max_workers=4,  # Monitoring doesn't need many workers
                thread_name_prefix="gpt-trader-monitor",
            ),
            ThreadPoolType.BACKGROUND: ThreadPoolConfig(
                pool_type=ThreadPoolType.BACKGROUND,
                max_workers=2,  # Background tasks are low priority
                thread_name_prefix="gpt-trader-bg",
            ),
        }

    def _initialize_thread_pools(self) -> None:
        """Initialize all thread pools"""
        with self.manager_lock:
            for pool_type, config in self.pool_configs.items():
                self.thread_pools[pool_type] = ManagedThreadPool(config, self)

            self.is_initialized = True

        logger.info(f"Initialized {len(self.thread_pools)} thread pools")

    def start(self) -> None:
        """Start concurrency manager and all background services"""
        if not self.is_initialized:
            raise ComponentException("Concurrency manager not initialized")

        # Start task scheduler
        self.task_scheduler.start()

        logger.info("Concurrency manager started")

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown concurrency manager and all resources"""
        logger.info("Shutting down concurrency manager...")

        self.shutdown_event.set()

        # Stop task scheduler
        self.task_scheduler.stop(timeout=timeout / 4)

        # Shutdown all thread pools
        for _pool_type, pool in self.thread_pools.items():
            pool.shutdown(wait=True, timeout=timeout / 4)

        logger.info("Concurrency manager shut down")

    def submit_task(
        self,
        pool_type: ThreadPoolType,
        function: Callable[..., T],
        *args,
        task_id: str | None = None,
        component_id: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> AsyncTaskResult[T]:
        """Submit task to appropriate thread pool"""

        if pool_type not in self.thread_pools:
            raise_validation_error(f"Unknown thread pool type: {pool_type}")

        pool = self.thread_pools[pool_type]
        return pool.submit_task(
            function=function,
            *args,
            priority=priority,
            task_id=task_id,
            component_id=component_id,
            **kwargs,
        )

    def submit_io_task(self, function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
        """Submit I/O-bound task"""
        return self.submit_task(ThreadPoolType.IO_BOUND, function, *args, **kwargs)

    def submit_cpu_task(self, function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
        """Submit CPU-bound task"""
        return self.submit_task(ThreadPoolType.CPU_BOUND, function, *args, **kwargs)

    def submit_monitoring_task(
        self, function: Callable[..., T], *args, **kwargs
    ) -> AsyncTaskResult[T]:
        """Submit monitoring task"""
        return self.submit_task(ThreadPoolType.MONITORING, function, *args, **kwargs)

    def submit_background_task(
        self, function: Callable[..., T], *args, **kwargs
    ) -> AsyncTaskResult[T]:
        """Submit background task"""
        return self.submit_task(ThreadPoolType.BACKGROUND, function, *args, **kwargs)

    def schedule_task(self, *args, **kwargs) -> str:
        """Schedule task for future execution"""
        return self.task_scheduler.schedule_task(*args, **kwargs)

    def schedule_recurring_task(self, *args, **kwargs) -> str:
        """Schedule recurring task"""
        return self.task_scheduler.schedule_recurring_task(*args, **kwargs)

    def cancel_scheduled_task(self, task_id: str) -> bool:
        """Cancel scheduled task"""
        return self.task_scheduler.cancel_task(task_id)

    def create_message_queue(self, queue_id: str, maxsize: int = 1000) -> MessageQueue:
        """Create message queue for inter-component communication"""
        with self.manager_lock:
            if queue_id in self.message_queues:
                return self.message_queues[queue_id]

            queue = MessageQueue(queue_id, maxsize)
            self.message_queues[queue_id] = queue
            return queue

    def get_message_queue(self, queue_id: str) -> MessageQueue | None:
        """Get existing message queue"""
        return self.message_queues.get(queue_id)

    def register_component(self, component: BaseComponent) -> None:
        """Register component with concurrency manager"""
        with self.manager_lock:
            self.component_registry[component.component_id] = weakref.ref(component)
        logger.debug(f"Registered component: {component.component_id}")

    def unregister_component(self, component_id: str) -> None:
        """Unregister component from concurrency manager"""
        with self.manager_lock:
            if component_id in self.component_registry:
                del self.component_registry[component_id]
        logger.debug(f"Unregistered component: {component_id}")

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics"""
        with self.manager_lock:
            pool_stats = {}
            for pool_type, pool in self.thread_pools.items():
                pool_stats[pool_type.value] = pool.get_pool_stats()

            queue_stats = {}
            for queue_id, queue in self.message_queues.items():
                queue_stats[queue_id] = queue.get_stats()

            # Active components
            active_components = []
            for component_id, component_ref in self.component_registry.items():
                component = component_ref()
                if component:
                    active_components.append(
                        {
                            "component_id": component_id,
                            "status": component.status.value,
                            "health": component.get_health_status().value,
                        }
                    )

            return {
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "thread_pools": pool_stats,
                "message_queues": queue_stats,
                "active_components": active_components,
                "scheduler_running": self.task_scheduler.is_running,
                "total_scheduled_tasks": len(self.task_scheduler.scheduled_tasks),
                "recurring_tasks": len(self.task_scheduler.recurring_tasks),
            }

    @contextmanager
    def component_scope(self, component: BaseComponent):
        """Context manager for component resource management"""
        self.register_component(component)
        try:
            yield
        finally:
            self.unregister_component(component.component_id)


# Global concurrency manager instance
_concurrency_manager: ConcurrencyManager | None = None
_manager_lock = threading.Lock()


def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager instance"""
    global _concurrency_manager

    with _manager_lock:
        if _concurrency_manager is None:
            _concurrency_manager = ConcurrencyManager()
            logger.info("Global concurrency manager created")

        return _concurrency_manager


def initialize_concurrency() -> ConcurrencyManager:
    """Initialize and start global concurrency manager"""
    manager = get_concurrency_manager()
    manager.start()
    return manager


def shutdown_concurrency(timeout: float = 30.0) -> None:
    """Shutdown global concurrency manager"""
    global _concurrency_manager

    with _manager_lock:
        if _concurrency_manager:
            _concurrency_manager.shutdown(timeout=timeout)
            _concurrency_manager = None
            logger.info("Global concurrency manager shut down")


# Convenience functions for common operations


def submit_io_task(function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
    """Submit I/O-bound task to global manager"""
    return get_concurrency_manager().submit_io_task(function, *args, **kwargs)


def submit_cpu_task(function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
    """Submit CPU-bound task to global manager"""
    return get_concurrency_manager().submit_cpu_task(function, *args, **kwargs)


def submit_monitoring_task(function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
    """Submit monitoring task to global manager"""
    return get_concurrency_manager().submit_monitoring_task(function, *args, **kwargs)


def submit_background_task(function: Callable[..., T], *args, **kwargs) -> AsyncTaskResult[T]:
    """Submit background task to global manager"""
    return get_concurrency_manager().submit_background_task(function, *args, **kwargs)


def schedule_task(task_id: str, function: Callable, *args, **kwargs) -> str:
    """Schedule task for future execution"""
    return get_concurrency_manager().schedule_task(task_id, function, *args, **kwargs)


def schedule_recurring_task(
    task_id: str, function: Callable, interval: timedelta, *args, **kwargs
) -> str:
    """Schedule recurring task"""
    return get_concurrency_manager().schedule_recurring_task(
        task_id, function, interval, *args, **kwargs
    )


def create_message_queue(queue_id: str, maxsize: int = 1000) -> MessageQueue:
    """Create message queue for inter-component communication"""
    return get_concurrency_manager().create_message_queue(queue_id, maxsize)
