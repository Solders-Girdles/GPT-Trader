"""
GPT-Trader Core Architecture Module

This module provides the foundational architecture components for the GPT-Trader system:
- Base classes for all components
- Common interfaces and protocols
- Shared utilities and patterns
- Configuration management
- Dependency injection framework
- Concurrency management
- Error handling framework

This core module standardizes patterns across all GPT-Trader components and provides
the architectural foundation for enterprise-grade trading operations.
"""

# Foundation Architecture
from .base import (
    BaseComponent,
    BaseEngine,
    BaseMonitor,
    BaseStrategy,
    ComponentConfig,
    ComponentStatus,
    HealthStatus,
)
from .concurrency import (
    AsyncTaskResult,
    ConcurrencyManager,
    IMessageHandler,
    MessageQueue,
    TaskPriority,
    ThreadPoolType,
    create_message_queue,
    get_concurrency_manager,
    initialize_concurrency,
    schedule_recurring_task,
    schedule_task,
    submit_background_task,
    submit_cpu_task,
    submit_io_task,
    submit_monitoring_task,
)
from .config import Environment, SystemConfig, get_config, initialize_config

# Component Integration
from .container import (
    ServiceContainer,
    ServiceLifetime,
    component,
    configure_services,
    get_container,
    injectable,
)
from .database import DatabaseManager, get_database, initialize_database
from .error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorManager,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    error_handling_context,
    get_error_manager,
    get_error_statistics,
    handle_errors,
    report_error,
    with_circuit_breaker,
    with_retry,
)
from .exceptions import (
    ComponentException,
    ConfigurationException,
    DatabaseException,
    DataException,
    ErrorCategory,
    ErrorSeverity,
    GPTTraderException,
    NetworkException,
    ResourceException,
    RiskException,
    TradingException,
    ValidationException,
)
from .migration import ArchitectureMigrationManager, MigrationPhase, MigrationStatus

__all__ = [
    # Base classes and foundation
    "BaseComponent",
    "BaseMonitor",
    "BaseEngine",
    "BaseStrategy",
    "ComponentStatus",
    "HealthStatus",
    "ComponentConfig",
    # Configuration
    "SystemConfig",
    "get_config",
    "initialize_config",
    "Environment",
    # Phase 1: Database
    "DatabaseManager",
    "get_database",
    "initialize_database",
    # Phase 1: Migration
    "ArchitectureMigrationManager",
    "MigrationStatus",
    "MigrationPhase",
    # Phase 1: Exceptions
    "GPTTraderException",
    "TradingException",
    "RiskException",
    "DataException",
    "ConfigurationException",
    "DatabaseException",
    "NetworkException",
    "ComponentException",
    "ResourceException",
    "ValidationException",
    "ErrorSeverity",
    "ErrorCategory",
    # Phase 2: Dependency Injection
    "ServiceContainer",
    "ServiceLifetime",
    "get_container",
    "configure_services",
    "injectable",
    "component",
    # Phase 2: Concurrency
    "ConcurrencyManager",
    "ThreadPoolType",
    "TaskPriority",
    "AsyncTaskResult",
    "MessageQueue",
    "IMessageHandler",
    "get_concurrency_manager",
    "initialize_concurrency",
    "submit_io_task",
    "submit_cpu_task",
    "submit_monitoring_task",
    "submit_background_task",
    "schedule_task",
    "schedule_recurring_task",
    "create_message_queue",
    # Phase 2: Error Handling
    "ErrorManager",
    "CircuitBreaker",
    "RetryHandler",
    "RetryConfig",
    "RetryStrategy",
    "CircuitBreakerConfig",
    "get_error_manager",
    "handle_errors",
    "with_circuit_breaker",
    "with_retry",
    "error_handling_context",
    "report_error",
    "get_error_statistics",
    # Phase 3: Performance & Observability
    "CacheManager",
    "IntelligentCache",
    "get_cache_manager",
    "get_cache",
    "cached",
    "cache_invalidate",
    "MetricsCollector",
    "MetricsRegistry",
    "get_metrics_collector",
    "get_metrics_registry",
    "track_execution_time",
    "count_calls",
    "PerformanceOptimizer",
    "PerformanceProfiler",
    "get_performance_optimizer",
    "create_profiler",
    "profile_performance",
    "ObservabilityEngine",
    "Alert",
    "AlertRule",
    "get_observability_engine",
    "create_alert",
    "start_trace",
    "trace_operation",
]

# Phase 3: Performance & Observability
from .analytics import (
    AnalyticsManager,
    ModelConfig,
    OptimizationObjective,
    create_performance_model,
    get_analytics_manager,
    optimize_latency,
    setup_anomaly_detection,
)
from .caching import (
    CacheManager,
    IntelligentCache,
    cache_invalidate,
    cached,
    get_cache,
    get_cache_manager,
)

# Phase 4: Operational Excellence
from .deployment import (
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentManager,
    DeploymentStrategy,
    canary_deploy,
    deploy_to_kubernetes,
    get_deployment_manager,
)
from .disaster_recovery import (
    DisasterRecoveryManager,
    FailoverConfiguration,
    RecoveryObjectives,
    configure_high_availability,
    create_scheduled_backup,
    get_disaster_recovery_manager,
    setup_database_replication,
)
from .metrics import (
    MetricsCollector,
    MetricsRegistry,
    count_calls,
    get_metrics_collector,
    get_metrics_registry,
    track_execution_time,
)
from .observability import (
    Alert,
    AlertRule,
    AlertSeverity,
    ObservabilityEngine,
    create_alert,
    get_observability_engine,
    start_trace,
    trace_operation,
)
from .performance import (
    PerformanceOptimizer,
    PerformanceProfiler,
    create_profiler,
    get_performance_optimizer,
    profile_performance,
)
from .security import (
    SecurityContext,
    SecurityManager,
    SecurityPrincipal,
    audit_operation,
    encrypt_sensitive_data,
    get_security_manager,
    require_authentication,
    require_authorization,
)

__all__.extend(
    [
        # Phase 3: Caching
        "CacheManager",
        "IntelligentCache",
        "get_cache_manager",
        "get_cache",
        "cached",
        "cache_invalidate",
        # Phase 3: Metrics
        "MetricsCollector",
        "MetricsRegistry",
        "get_metrics_collector",
        "get_metrics_registry",
        "track_execution_time",
        "count_calls",
        # Phase 3: Performance
        "PerformanceOptimizer",
        "PerformanceProfiler",
        "get_performance_optimizer",
        "create_profiler",
        "profile_performance",
        # Phase 3: Observability
        "ObservabilityEngine",
        "Alert",
        "AlertRule",
        "AlertSeverity",
        "get_observability_engine",
        "create_alert",
        "start_trace",
        "trace_operation",
        # Phase 4: Deployment
        "DeploymentManager",
        "DeploymentConfig",
        "DeploymentStrategy",
        "DeploymentEnvironment",
        "get_deployment_manager",
        "deploy_to_kubernetes",
        "canary_deploy",
        # Phase 4: Security
        "SecurityManager",
        "SecurityPrincipal",
        "SecurityContext",
        "get_security_manager",
        "require_authentication",
        "require_authorization",
        "encrypt_sensitive_data",
        "audit_operation",
        # Phase 4: Disaster Recovery
        "DisasterRecoveryManager",
        "RecoveryObjectives",
        "FailoverConfiguration",
        "get_disaster_recovery_manager",
        "configure_high_availability",
        "setup_database_replication",
        "create_scheduled_backup",
        # Phase 4: Analytics
        "AnalyticsManager",
        "ModelConfig",
        "OptimizationObjective",
        "get_analytics_manager",
        "create_performance_model",
        "setup_anomaly_detection",
        "optimize_latency",
    ]
)

__version__ = "4.0.0"
