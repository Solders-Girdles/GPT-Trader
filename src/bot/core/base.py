"""
GPT-Trader Base Classes and Interfaces

Foundation architecture providing standardized base classes for all GPT-Trader components:
- BaseComponent: Core component lifecycle and interface
- BaseMonitor: Monitoring and alerting components
- BaseEngine: Execution and processing engines
- BaseStrategy: Trading strategy implementations

All GPT-Trader components should inherit from these base classes to ensure
consistent behavior, interfaces, and integration patterns.
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from .exceptions import ComponentException, ConfigurationException, raise_config_error


class ComponentStatus(Enum):
    """Component lifecycle status"""

    CREATED = "created"  # Component instantiated but not initialized
    INITIALIZING = "initializing"  # Component is being initialized
    INITIALIZED = "initialized"  # Component initialized but not started
    STARTING = "starting"  # Component is starting up
    RUNNING = "running"  # Component is running normally
    STOPPING = "stopping"  # Component is shutting down
    STOPPED = "stopped"  # Component has stopped
    ERROR = "error"  # Component is in error state
    FAILED = "failed"  # Component has failed permanently


class HealthStatus(Enum):
    """Component health status"""

    HEALTHY = "healthy"  # Component is operating normally
    DEGRADED = "degraded"  # Component is operational but with issues
    UNHEALTHY = "unhealthy"  # Component has significant issues
    CRITICAL = "critical"  # Component is in critical state
    UNKNOWN = "unknown"  # Health status cannot be determined


@dataclass
class ComponentMetrics:
    """Standard metrics collected for all components"""

    component_id: str
    component_type: str

    # Performance metrics
    uptime: timedelta = field(default_factory=lambda: timedelta())
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

    # Operational metrics
    operations_total: int = 0
    operations_successful: int = 0
    operations_failed: int = 0

    # Error tracking
    error_count: int = 0
    last_error_time: datetime | None = None
    last_error_message: str | None = None

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate operation success rate"""
        if self.operations_total == 0:
            return 0.0
        return self.operations_successful / self.operations_total

    @property
    def error_rate(self) -> float:
        """Calculate operation error rate"""
        if self.operations_total == 0:
            return 0.0
        return self.operations_failed / self.operations_total


@dataclass
class ComponentConfig:
    """Base configuration for all components"""

    component_id: str
    component_type: str

    # Directory configuration
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    config_dir: Path = field(default_factory=lambda: Path("config"))

    # Operational configuration
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))

    # Integration configuration
    database_enabled: bool = True
    monitoring_enabled: bool = True
    alerting_enabled: bool = True

    # Custom configuration
    custom_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration"""
        # Ensure directories are Path objects
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)
        if isinstance(self.config_dir, str):
            self.config_dir = Path(self.config_dir)

        # Validate component identifiers
        if not self.component_id:
            raise_config_error("Component ID cannot be empty")
        if not self.component_type:
            raise_config_error("Component type cannot be empty")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise_config_error(f"Invalid log level: {self.log_level}")

    def get_component_data_dir(self) -> Path:
        """Get component-specific data directory"""
        return self.data_dir / self.component_type / self.component_id

    def get_component_logs_dir(self) -> Path:
        """Get component-specific logs directory"""
        return self.logs_dir / self.component_type / self.component_id


T = TypeVar("T", bound=ComponentConfig)


class BaseComponent(Generic[T], ABC):
    """
    Base class for all GPT-Trader components

    Provides standardized lifecycle management, configuration handling,
    logging, metrics collection, and health monitoring for all components.
    """

    def __init__(self, config: T) -> None:
        """Initialize base component"""
        # Validate configuration
        if not isinstance(config, ComponentConfig):
            raise ConfigurationException(
                f"Config must be ComponentConfig instance, got {type(config)}"
            )

        self.config = config
        self.component_id = config.component_id
        self.component_type = config.component_type

        # Component state
        self.status = ComponentStatus.CREATED
        self.health_status = HealthStatus.UNKNOWN
        self.started_at: datetime | None = None
        self.stopped_at: datetime | None = None

        # Threading and lifecycle
        self._shutdown_event = threading.Event()
        self._health_check_thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()

        # Initialize logging
        self._setup_logging()

        # Initialize metrics
        self.metrics = ComponentMetrics(
            component_id=self.component_id, component_type=self.component_type
        )

        # Create directories
        self._create_directories()

        # Component-specific initialization
        try:
            self.status = ComponentStatus.INITIALIZING
            self._initialize_component()
            self.status = ComponentStatus.INITIALIZED
            self.logger.info(f"Component {self.component_id} initialized successfully")
        except Exception as e:
            self.status = ComponentStatus.FAILED
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise ComponentException(
                f"Failed to initialize component {self.component_id}",
                component=self.component_id,
                context={"error": str(e)},
            )

    def _setup_logging(self) -> None:
        """Setup component-specific logging"""
        self.logger = logging.getLogger(f"{self.component_type}.{self.component_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Create logs directory
        logs_dir = self.config.get_component_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler if not already present
        if not self.logger.handlers:
            file_handler = logging.FileHandler(logs_dir / f"{self.component_id}.log")
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _create_directories(self) -> None:
        """Create necessary directories"""
        dirs_to_create = [
            self.config.get_component_data_dir(),
            self.config.get_component_logs_dir(),
        ]

        for directory in dirs_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationException(
                    f"Failed to create directory {directory}",
                    component=self.component_id,
                    context={"directory": str(directory), "error": str(e)},
                )

    @abstractmethod
    def _initialize_component(self):
        """Component-specific initialization logic"""
        pass

    @abstractmethod
    def _start_component(self):
        """Component-specific startup logic"""
        pass

    @abstractmethod
    def _stop_component(self):
        """Component-specific shutdown logic"""
        pass

    @abstractmethod
    def _health_check(self) -> HealthStatus:
        """Component-specific health check logic"""
        pass

    def start(self) -> None:
        """Start the component"""
        with self._lifecycle_lock:
            if self.status not in [ComponentStatus.INITIALIZED, ComponentStatus.STOPPED]:
                raise ComponentException(
                    f"Cannot start component in {self.status.value} status",
                    component=self.component_id,
                    context={"current_status": self.status.value},
                )

            try:
                self.status = ComponentStatus.STARTING
                self.logger.info(f"Starting component {self.component_id}")

                # Clear shutdown event
                self._shutdown_event.clear()

                # Start component-specific logic
                self._start_component()

                # Start health check thread if enabled
                if self.config.enable_health_checks:
                    self._start_health_monitoring()

                # Update status and timing
                self.status = ComponentStatus.RUNNING
                self.started_at = datetime.now()
                self.health_status = HealthStatus.HEALTHY

                self.logger.info(f"Component {self.component_id} started successfully")

            except Exception as e:
                self.status = ComponentStatus.ERROR
                self.logger.error(f"Failed to start component: {str(e)}")
                raise ComponentException(
                    f"Failed to start component {self.component_id}",
                    component=self.component_id,
                    context={"error": str(e)},
                )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the component gracefully"""
        with self._lifecycle_lock:
            if self.status not in [ComponentStatus.RUNNING, ComponentStatus.ERROR]:
                self.logger.warning(
                    f"Component already stopped or stopping (status: {self.status.value})"
                )
                return

            try:
                self.status = ComponentStatus.STOPPING
                self.logger.info(f"Stopping component {self.component_id}")

                # Signal shutdown
                self._shutdown_event.set()

                # Stop health monitoring
                if self._health_check_thread and self._health_check_thread.is_alive():
                    self._health_check_thread.join(timeout=5.0)

                # Stop component-specific logic
                self._stop_component()

                # Update status and timing
                self.status = ComponentStatus.STOPPED
                self.stopped_at = datetime.now()
                self.health_status = HealthStatus.UNKNOWN

                self.logger.info(f"Component {self.component_id} stopped successfully")

            except Exception as e:
                self.status = ComponentStatus.ERROR
                self.logger.error(f"Error during component shutdown: {str(e)}")
                raise ComponentException(
                    f"Failed to stop component {self.component_id}",
                    component=self.component_id,
                    context={"error": str(e)},
                )

    def _start_health_monitoring(self) -> None:
        """Start health check monitoring thread"""
        self._health_check_thread = threading.Thread(
            target=self._health_monitoring_loop, name=f"{self.component_id}-health", daemon=True
        )
        self._health_check_thread.start()

    def _health_monitoring_loop(self) -> None:
        """Health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Perform health check
                self.health_status = self._health_check()

                # Update uptime
                if self.started_at:
                    self.metrics.uptime = datetime.now() - self.started_at

                # Update timestamp
                self.metrics.last_updated = datetime.now()

            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                self.health_status = HealthStatus.CRITICAL
                self.metrics.error_count += 1
                self.metrics.last_error_time = datetime.now()
                self.metrics.last_error_message = str(e)

            # Wait for next check or shutdown
            if self._shutdown_event.wait(self.config.health_check_interval.total_seconds()):
                break

    def get_status(self) -> dict[str, Any]:
        """Get current component status"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "status": self.status.value,
            "health_status": self.health_status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "uptime_seconds": self.metrics.uptime.total_seconds(),
            "metrics": {
                "operations_total": self.metrics.operations_total,
                "success_rate": self.metrics.success_rate,
                "error_count": self.metrics.error_count,
                "last_error_time": (
                    self.metrics.last_error_time.isoformat()
                    if self.metrics.last_error_time
                    else None
                ),
            },
        }

    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        return self.health_status

    def get_metrics(self) -> ComponentMetrics:
        """Get current metrics"""
        return self.metrics

    def is_running(self) -> bool:
        """Check if component is running"""
        return self.status == ComponentStatus.RUNNING

    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def record_operation(self, success: bool = True, error_message: str | None = None) -> None:
        """Record an operation for metrics tracking"""
        self.metrics.operations_total += 1

        if success:
            self.metrics.operations_successful += 1
        else:
            self.metrics.operations_failed += 1
            self.metrics.error_count += 1
            if error_message:
                self.metrics.last_error_time = datetime.now()
                self.metrics.last_error_message = error_message

    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric value"""
        self.metrics.custom_metrics[name] = value

    def get_custom_metric(self, name: str, default: Any = None) -> Any:
        """Get a custom metric value"""
        return self.metrics.custom_metrics.get(name, default)

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class BaseMonitor(BaseComponent):
    """
    Base class for monitoring components

    Specialized base class for components that monitor system state,
    collect metrics, and generate alerts.
    """

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)

        # Monitoring-specific attributes
        self.monitoring_interval: timedelta = timedelta(seconds=30)
        self.alert_callbacks: list[Callable] = []
        self.monitored_components: dict[str, BaseComponent] = {}

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def register_component(self, component: BaseComponent) -> None:
        """Register a component for monitoring"""
        self.monitored_components[component.component_id] = component
        self.logger.info(f"Registered component for monitoring: {component.component_id}")

    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from monitoring"""
        if component_id in self.monitored_components:
            del self.monitored_components[component_id]
            self.logger.info(f"Unregistered component from monitoring: {component_id}")

    @abstractmethod
    def _collect_metrics(self) -> dict[str, Any]:
        """Collect monitoring metrics"""
        pass

    @abstractmethod
    def _evaluate_alerts(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate alert conditions based on metrics"""
        pass


class BaseEngine(BaseComponent):
    """
    Base class for execution engines

    Specialized base class for components that execute trading operations,
    process orders, and manage execution state.
    """

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)

        # Engine-specific attributes
        self.execution_queue: Any | None = None  # Will be implemented by subclasses
        self.processing_stats = {
            "items_processed": 0,
            "items_queued": 0,
            "processing_rate": 0.0,
            "last_processing_time": None,
        }

    @abstractmethod
    def _process_execution_request(self, request: Any) -> Any:
        """Process a single execution request"""
        pass

    @abstractmethod
    def _validate_execution_request(self, request: Any) -> bool:
        """Validate execution request before processing"""
        pass

    def get_processing_stats(self) -> dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()


class BaseStrategy(BaseComponent):
    """
    Base class for trading strategies

    Specialized base class for trading strategy implementations providing
    standardized interfaces for signal generation and position management.
    """

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)

        # Strategy-specific attributes
        self.strategy_parameters: dict[str, Any] = {}
        self.performance_metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
        }

    @abstractmethod
    def _generate_signals(self, market_data: Any) -> list[Any]:
        """Generate trading signals based on market data"""
        pass

    @abstractmethod
    def _calculate_position_size(self, signal: Any, portfolio_state: Any) -> float:
        """Calculate position size for a trading signal"""
        pass

    def get_strategy_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters"""
        return self.strategy_parameters.copy()

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def update_performance_metric(self, metric_name: str, value: Any) -> None:
        """Update a performance metric"""
        self.performance_metrics[metric_name] = value
