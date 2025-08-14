"""
GPT-Trader Unified Monitoring System
Consolidated monitoring, alerting, and health checking
"""

# Import existing monitoring components (may have import issues due to existing codebase)
try:
    from .monitor import UnifiedMonitor
    from .alerts import AlertManager
    from .metrics import MetricsCollector
    from .health import HealthChecker
    LEGACY_MONITORING_AVAILABLE = True
except ImportError:
    LEGACY_MONITORING_AVAILABLE = False

# Import new Phase 3 Week 7 structured logging system
try:
    from .structured_logger import (
        EnhancedStructuredLogger,
        get_logger,
        configure_logging,
        LogFormat,
        SpanType,
        traced_operation
    )
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False

# Import ML logging integration
try:
    from .ml_logging_integration import (
        MLLoggingMixin,
        log_ml_operation,
        log_data_quality_check,
        log_model_deployment,
        log_prediction_batch
    )
    ML_LOGGING_AVAILABLE = True
except ImportError:
    ML_LOGGING_AVAILABLE = False

# Define exports based on what's available
__all__ = []

if LEGACY_MONITORING_AVAILABLE:
    __all__.extend([
        'UnifiedMonitor',
        'AlertManager', 
        'MetricsCollector',
        'HealthChecker'
    ])

if STRUCTURED_LOGGING_AVAILABLE:
    __all__.extend([
        'EnhancedStructuredLogger',
        'get_logger',
        'configure_logging',
        'LogFormat',
        'SpanType',
        'traced_operation'
    ])

if ML_LOGGING_AVAILABLE:
    __all__.extend([
        'MLLoggingMixin',
        'log_ml_operation',
        'log_data_quality_check',
        'log_model_deployment',
        'log_prediction_batch'
    ])