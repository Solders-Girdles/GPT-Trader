"""
Retraining Scheduler for GPT-Trader
Phase 3, Week 5-6: ADAPT-010, ADAPT-015

Advanced scheduling system for automated retraining with intelligent
scheduling, resource optimization, and priority management.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import heapq
import cron_descriptor
from croniter import croniter
import pytz

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of retraining schedules"""
    CRON = "cron"                    # Cron expression
    INTERVAL = "interval"            # Fixed interval
    PERFORMANCE = "performance"      # Performance-based
    DRIFT = "drift"                 # Drift-based
    MANUAL = "manual"               # Manual trigger
    ADAPTIVE = "adaptive"           # Adaptive scheduling


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 3
    HIGH = 7
    URGENT = 9
    EMERGENCY = 10


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled task"""
    task_id: str
    schedule_type: ScheduleType
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Cron scheduling
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    
    # Interval scheduling
    interval_minutes: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Performance-based scheduling
    performance_threshold: Optional[float] = None
    performance_window: Optional[int] = None
    
    # Resource constraints
    max_execution_time: int = 7200  # 2 hours
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    requires_approval: bool = False
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_minutes: int = 30
    exponential_backoff: bool = True
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Execution window
    allowed_hours: Optional[List[int]] = None  # [0-23]
    blocked_hours: Optional[List[int]] = None
    allowed_days: Optional[List[int]] = None   # [0-6] Monday=0
    
    # Adaptive parameters
    min_interval_hours: int = 6
    max_interval_hours: int = 168  # 1 week
    success_factor: float = 1.2
    failure_factor: float = 0.8
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class ScheduledTask:
    """A scheduled retraining task"""
    task_id: str
    config: ScheduleConfig
    callback: Callable[..., Any]
    next_run: datetime
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Retry tracking
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    
    # Dependencies
    dependencies_met: bool = False
    blocking_tasks: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Compare tasks for priority queue"""
        if self.config.priority.value \!= other.config.priority.value:
            return self.config.priority.value > other.config.priority.value
        return self.next_run < other.next_run


@dataclass
class ExecutionResult:
    """Result of task execution"""
    task_id: str
    success: bool
    started_at: datetime
    completed_at: datetime
    execution_time: float
    result: Optional[Any] = None
    error: Optional[str] = None
    resource_usage: Optional[Dict[str, float]] = None


class RetrainingScheduler:
    """
    Advanced retraining scheduler with intelligent scheduling capabilities.
    
    Features:
    - Multiple scheduling types (cron, interval, performance-based)
    - Priority-based task queue
    - Resource constraint management
    - Dependency tracking
    - Adaptive scheduling
    - Retry mechanisms
    - Execution window controls
    """
    
    def __init__(self, timezone: str = "UTC"):
        """Initialize retraining scheduler
        
        Args:
            timezone: Default timezone for scheduling
        """
        self.timezone = pytz.timezone(timezone)
        
        # Task management
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.task_queue: List[ScheduledTask] = []  # Priority queue
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[ExecutionResult] = []
        
        # State management
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.executor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.execution_stats: Dict[str, Dict[str, float]] = {}
        
        # Resource monitoring
        self.resource_usage: Dict[str, float] = {}
        self.max_concurrent_tasks = 3
        
        logger.info("Initialized retraining scheduler")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="RetrainingScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        # Start executor thread
        self.executor_thread = threading.Thread(
            target=self._executor_loop,
            name="TaskExecutor",
            daemon=True
        )
        self.executor_thread.start()
        
        logger.info("Started retraining scheduler")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        logger.info("Stopping retraining scheduler")
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        if self.executor_thread and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=10)
        
        logger.info("Stopped retraining scheduler")
    
    def add_schedule(self, 
                     config: ScheduleConfig, 
                     callback: Callable[..., Any]) -> bool:
        """Add a new schedule
        
        Args:
            config: Schedule configuration
            callback: Function to call when scheduled
            
        Returns:
            True if schedule was added successfully
        """
        try:
            # Validate configuration
            self._validate_schedule_config(config)
            
            # Store schedule
            self.schedules[config.task_id] = config
            
            # Calculate next run time
            next_run = self._calculate_next_run(config)
            
            if next_run:
                # Create scheduled task
                task = ScheduledTask(
                    task_id=config.task_id,
                    config=config,
                    callback=callback,
                    next_run=next_run,
                    scheduled_at=datetime.now()
                )
                
                # Add to queue
                heapq.heappush(self.task_queue, task)
                
                logger.info(f"Added schedule {config.task_id}: next run at {next_run}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to add schedule {config.task_id}: {e}")
            return False
    
    def remove_schedule(self, task_id: str) -> bool:
        """Remove a schedule
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if schedule was removed
        """
        if task_id in self.schedules:
            del self.schedules[task_id]
            
            # Remove from queue (inefficient but simple)
            self.task_queue = [task for task in self.task_queue if task.task_id \!= task_id]
            heapq.heapify(self.task_queue)
            
            logger.info(f"Removed schedule {task_id}")
            return True
        
        return False
    
    def update_schedule(self, 
                        task_id: str, 
                        config: ScheduleConfig) -> bool:
        """Update an existing schedule
        
        Args:
            task_id: Task identifier
            config: New configuration
            
        Returns:
            True if schedule was updated
        """
        if task_id in self.schedules:
            old_callback = None
            
            # Find existing callback
            for task in self.task_queue:
                if task.task_id == task_id:
                    old_callback = task.callback
                    break
            
            if old_callback:
                self.remove_schedule(task_id)
                return self.add_schedule(config, old_callback)
        
        return False
    
    def trigger_immediate(self, 
                          task_id: str, 
                          priority: TaskPriority = TaskPriority.HIGH) -> bool:
        """Trigger immediate execution of a task
        
        Args:
            task_id: Task identifier
            priority: Execution priority
            
        Returns:
            True if task was triggered
        """
        if task_id not in self.schedules:
            logger.error(f"Schedule {task_id} not found")
            return False
        
        config = self.schedules[task_id]
        
        # Find callback
        callback = None
        for task in self.task_queue:
            if task.task_id == task_id:
                callback = task.callback
                break
        
        if not callback:
            logger.error(f"Callback not found for task {task_id}")
            return False
        
        # Create immediate task
        immediate_config = ScheduleConfig(
            task_id=f"{task_id}_immediate_{int(datetime.now().timestamp())}",
            schedule_type=ScheduleType.MANUAL,
            name=f"Immediate {config.name}",
            description=f"Immediate execution of {config.description}",
            priority=priority
        )
        
        task = ScheduledTask(
            task_id=immediate_config.task_id,
            config=immediate_config,
            callback=callback,
            next_run=datetime.now()
        )
        
        heapq.heappush(self.task_queue, task)
        logger.info(f"Triggered immediate execution of {task_id}")
        return True
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Started scheduler loop")
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for ready tasks
                ready_tasks = []
                while (self.task_queue and 
                       self.task_queue[0].next_run <= current_time):
                    task = heapq.heappop(self.task_queue)
                    ready_tasks.append(task)
                
                # Process ready tasks
                for task in ready_tasks:
                    if self._can_execute_task(task):
                        task.status = TaskStatus.SCHEDULED
                        self.running_tasks[task.task_id] = task
                        logger.info(f"Scheduled task {task.task_id} for execution")
                    else:
                        # Reschedule for later
                        task.next_run = current_time + timedelta(minutes=5)
                        heapq.heappush(self.task_queue, task)
                
                # Reschedule recurring tasks
                self._reschedule_recurring_tasks()
                
                # Adaptive scheduling adjustments
                self._adjust_adaptive_schedules()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
        
        logger.info("Stopped scheduler loop")
    
    def _executor_loop(self):
        """Main executor loop"""
        logger.info("Started executor loop")
        
        while not self.stop_event.is_set():
            try:
                # Get tasks ready for execution
                executable_tasks = [
                    task for task in self.running_tasks.values()
                    if task.status == TaskStatus.SCHEDULED
                ]
                
                # Sort by priority
                executable_tasks.sort(key=lambda x: x.config.priority.value, reverse=True)
                
                # Execute tasks (up to max concurrent)
                current_running = len([
                    task for task in self.running_tasks.values()
                    if task.status == TaskStatus.RUNNING
                ])
                
                available_slots = self.max_concurrent_tasks - current_running
                
                for task in executable_tasks[:available_slots]:
                    if self._check_dependencies(task) and self._check_execution_window(task):
                        self._execute_task(task)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                time.sleep(30)
        
        logger.info("Stopped executor loop")
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        def execute_in_thread():
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                logger.info(f"Executing task {task.task_id}")
                
                # Execute callback
                start_time = time.time()
                result = task.callback()
                end_time = time.time()
                
                # Record success
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.execution_time = end_time - start_time
                task.result = result
                
                # Update performance history
                self._update_performance_history(task.task_id, True, task.execution_time)
                
                # Create execution result
                exec_result = ExecutionResult(
                    task_id=task.task_id,
                    success=True,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    execution_time=task.execution_time,
                    result=result
                )
                
                self.completed_tasks.append(exec_result)
                
                logger.info(f"Task {task.task_id} completed successfully in {task.execution_time:.1f}s")
            
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = str(e)
                
                # Update performance history
                self._update_performance_history(task.task_id, False, 0)
                
                # Handle retry
                if task.retry_count < task.config.max_retries:
                    self._schedule_retry(task)
                else:
                    # Create failure result
                    exec_result = ExecutionResult(
                        task_id=task.task_id,
                        success=False,
                        started_at=task.started_at,
                        completed_at=task.completed_at,
                        execution_time=0,
                        error=str(e)
                    )
                    
                    self.completed_tasks.append(exec_result)
            
            finally:
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
        
        # Start execution in separate thread
        thread = threading.Thread(
            target=execute_in_thread,
            name=f"TaskExec_{task.task_id}",
            daemon=True
        )
        thread.start()
    
    def _schedule_retry(self, task: ScheduledTask):
        """Schedule task retry"""
        task.retry_count += 1
        task.last_retry_at = datetime.now()
        
        # Calculate retry delay
        delay_minutes = task.config.retry_delay_minutes
        if task.config.exponential_backoff:
            delay_minutes *= (2 ** (task.retry_count - 1))
        
        task.next_run = datetime.now() + timedelta(minutes=delay_minutes)
        task.status = TaskStatus.PENDING
        
        heapq.heappush(self.task_queue, task)
        
        logger.info(f"Scheduled retry {task.retry_count} for task {task.task_id} in {delay_minutes} minutes")
    
    def _can_execute_task(self, task: ScheduledTask) -> bool:
        """Check if task can be executed"""
        # Check resource constraints
        if not self._check_resource_availability(task):
            return False
        
        # Check concurrent limit
        current_running = len([
            t for t in self.running_tasks.values()
            if t.status == TaskStatus.RUNNING
        ])
        
        if current_running >= self.max_concurrent_tasks:
            return False
        
        # Check approval requirement
        if task.config.requires_approval:
            # In real implementation, check approval status
            pass
        
        return True
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are met"""
        for dep_id in task.config.depends_on:
            # Check if dependency completed successfully
            dep_completed = False
            for result in self.completed_tasks:
                if (result.task_id.startswith(dep_id) and 
                    result.success and 
                    result.completed_at > task.created_at):
                    dep_completed = True
                    break
            
            if not dep_completed:
                return False
        
        return True
    
    def _check_execution_window(self, task: ScheduledTask) -> bool:
        """Check if current time is within execution window"""
        now = datetime.now()
        
        # Check allowed hours
        if task.config.allowed_hours:
            if now.hour not in task.config.allowed_hours:
                return False
        
        # Check blocked hours
        if task.config.blocked_hours:
            if now.hour in task.config.blocked_hours:
                return False
        
        # Check allowed days
        if task.config.allowed_days:
            if now.weekday() not in task.config.allowed_days:
                return False
        
        return True
    
    def _check_resource_availability(self, task: ScheduledTask) -> bool:
        """Check if sufficient resources are available"""
        # Check memory (simplified)
        import psutil
        
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < task.config.max_memory_gb:
            return False
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > task.config.max_cpu_percent:
            return False
        
        return True
    
    def _calculate_next_run(self, config: ScheduleConfig) -> Optional[datetime]:
        """Calculate next run time for a schedule"""
        now = datetime.now(self.timezone)
        
        if config.schedule_type == ScheduleType.CRON:
            if not config.cron_expression:
                return None
            
            try:
                cron = croniter(config.cron_expression, now)
                return cron.get_next(datetime)
            except Exception as e:
                logger.error(f"Invalid cron expression {config.cron_expression}: {e}")
                return None
        
        elif config.schedule_type == ScheduleType.INTERVAL:
            if not config.interval_minutes:
                return None
            
            next_run = now + timedelta(minutes=config.interval_minutes)
            
            # Check time constraints
            if config.start_time and next_run < config.start_time:
                next_run = config.start_time
            
            if config.end_time and next_run > config.end_time:
                return None
            
            return next_run
        
        elif config.schedule_type == ScheduleType.MANUAL:
            return now  # Execute immediately
        
        elif config.schedule_type == ScheduleType.ADAPTIVE:
            # Adaptive scheduling based on performance
            base_interval = config.min_interval_hours
            
            if config.task_id in self.performance_history:
                performance = self.performance_history[config.task_id]
                if performance:
                    success_rate = sum(performance[-10:]) / len(performance[-10:])
                    
                    if success_rate > 0.8:
                        # High success rate, can wait longer
                        base_interval *= config.success_factor
                    elif success_rate < 0.5:
                        # Low success rate, schedule sooner
                        base_interval *= config.failure_factor
            
            # Ensure within bounds
            base_interval = max(config.min_interval_hours, 
                               min(base_interval, config.max_interval_hours))
            
            return now + timedelta(hours=base_interval)
        
        return None
    
    def _reschedule_recurring_tasks(self):
        """Reschedule recurring tasks"""
        for task_id, config in self.schedules.items():
            if config.schedule_type in [ScheduleType.CRON, ScheduleType.INTERVAL, ScheduleType.ADAPTIVE]:
                # Check if task needs to be rescheduled
                has_pending = any(task.task_id == task_id and task.status == TaskStatus.PENDING 
                                 for task in self.task_queue)
                
                has_running = task_id in self.running_tasks
                
                if not has_pending and not has_running:
                    # Find callback
                    callback = None
                    for completed in self.completed_tasks:
                        if completed.task_id.startswith(task_id):
                            # Get original callback (this is simplified)
                            break
                    
                    if callback is None:
                        continue
                    
                    # Calculate next run
                    next_run = self._calculate_next_run(config)
                    
                    if next_run:
                        new_task = ScheduledTask(
                            task_id=f"{task_id}_{int(datetime.now().timestamp())}",
                            config=config,
                            callback=callback,
                            next_run=next_run
                        )
                        
                        heapq.heappush(self.task_queue, new_task)
    
    def _adjust_adaptive_schedules(self):
        """Adjust adaptive schedules based on performance"""
        for task_id, config in self.schedules.items():
            if config.schedule_type == ScheduleType.ADAPTIVE:
                # Update adaptive parameters based on performance
                if task_id in self.performance_history:
                    performance = self.performance_history[task_id]
                    if len(performance) >= 5:
                        recent_success = sum(performance[-5:]) / 5
                        
                        # Adjust success/failure factors
                        if recent_success > 0.9:
                            config.success_factor = min(2.0, config.success_factor * 1.1)
                        elif recent_success < 0.3:
                            config.failure_factor = max(0.5, config.failure_factor * 0.9)
    
    def _update_performance_history(self, task_id: str, success: bool, execution_time: float):
        """Update performance history for a task"""
        if task_id not in self.performance_history:
            self.performance_history[task_id] = []
        
        self.performance_history[task_id].append(1.0 if success else 0.0)
        
        # Keep only recent history
        if len(self.performance_history[task_id]) > 100:
            self.performance_history[task_id] = self.performance_history[task_id][-100:]
        
        # Update execution stats
        if task_id not in self.execution_stats:
            self.execution_stats[task_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0
            }
        
        stats = self.execution_stats[task_id]
        stats["total_executions"] += 1
        if success:
            stats["successful_executions"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_executions"]
    
    def _validate_schedule_config(self, config: ScheduleConfig):
        """Validate schedule configuration"""
        if config.schedule_type == ScheduleType.CRON:
            if not config.cron_expression:
                raise ValueError("Cron expression required for CRON schedule type")
            
            # Validate cron expression
            try:
                croniter(config.cron_expression)
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {e}")
        
        elif config.schedule_type == ScheduleType.INTERVAL:
            if not config.interval_minutes or config.interval_minutes <= 0:
                raise ValueError("Valid interval_minutes required for INTERVAL schedule type")
    
    # Public API methods
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "is_running": self.is_running,
            "total_schedules": len(self.schedules),
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "next_task": self.task_queue[0].next_run.isoformat() if self.task_queue else None
        }
    
    def get_task_history(self, task_id: str) -> List[ExecutionResult]:
        """Get execution history for a task"""
        return [result for result in self.completed_tasks 
                if result.task_id.startswith(task_id)]
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all tasks"""
        summary = {}
        
        for task_id, stats in self.execution_stats.items():
            success_rate = (stats["successful_executions"] / stats["total_executions"] 
                           if stats["total_executions"] > 0 else 0.0)
            
            summary[task_id] = {
                "success_rate": success_rate,
                "avg_execution_time": stats["avg_time"],
                "total_executions": stats["total_executions"]
            }
        
        return summary


# Helper functions
def create_cron_schedule(task_id: str,
                        name: str,
                        cron_expression: str,
                        priority: TaskPriority = TaskPriority.MEDIUM,
                        **kwargs) -> ScheduleConfig:
    """Create a cron-based schedule"""
    return ScheduleConfig(
        task_id=task_id,
        schedule_type=ScheduleType.CRON,
        name=name,
        description=f"Cron schedule: {cron_expression}",
        priority=priority,
        cron_expression=cron_expression,
        **kwargs
    )


def create_interval_schedule(task_id: str,
                           name: str,
                           interval_minutes: int,
                           priority: TaskPriority = TaskPriority.MEDIUM,
                           **kwargs) -> ScheduleConfig:
    """Create an interval-based schedule"""
    return ScheduleConfig(
        task_id=task_id,
        schedule_type=ScheduleType.INTERVAL,
        name=name,
        description=f"Interval schedule: every {interval_minutes} minutes",
        priority=priority,
        interval_minutes=interval_minutes,
        **kwargs
    )


def create_adaptive_schedule(task_id: str,
                           name: str,
                           min_interval_hours: int = 6,
                           max_interval_hours: int = 168,
                           priority: TaskPriority = TaskPriority.MEDIUM,
                           **kwargs) -> ScheduleConfig:
    """Create an adaptive schedule"""
    return ScheduleConfig(
        task_id=task_id,
        schedule_type=ScheduleType.ADAPTIVE,
        name=name,
        description=f"Adaptive schedule: {min_interval_hours}-{max_interval_hours} hours",
        priority=priority,
        min_interval_hours=min_interval_hours,
        max_interval_hours=max_interval_hours,
        **kwargs
    )


# Predefined schedules
DAILY_RETRAINING_SCHEDULE = create_cron_schedule(
    task_id="daily_retraining",
    name="Daily Model Retraining",
    cron_expression="0 2 * * *",  # 2 AM daily
    priority=TaskPriority.HIGH,
    max_execution_time=7200,  # 2 hours
    allowed_hours=[1, 2, 3, 4, 5]  # Early morning hours
)

WEEKLY_RETRAINING_SCHEDULE = create_cron_schedule(
    task_id="weekly_retraining",
    name="Weekly Model Retraining",
    cron_expression="0 1 * * 0",  # 1 AM Sunday
    priority=TaskPriority.MEDIUM,
    max_execution_time=14400,  # 4 hours
    allowed_days=[6]  # Sunday only
)

PERFORMANCE_ADAPTIVE_SCHEDULE = create_adaptive_schedule(
    task_id="performance_adaptive",
    name="Performance-Based Adaptive Retraining",
    min_interval_hours=6,
    max_interval_hours=72,
    priority=TaskPriority.HIGH
)
EOF < /dev/null