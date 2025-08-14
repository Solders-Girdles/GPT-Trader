"""
GPT-Trader Exception Hierarchy

Standardized exception classes for all GPT-Trader components providing:
- Consistent error categorization
- Structured error information
- Error recovery guidance
- Logging and monitoring integration

All GPT-Trader components should use these exceptions rather than generic
Exception classes to enable proper error handling and recovery.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Never


class ErrorSeverity(Enum):
    """Error severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error category classification"""

    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    EXECUTION = "execution"
    NETWORK = "network"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"


class GPTTraderException(Exception):
    """
    Base exception for all GPT-Trader errors

    Provides structured error information including:
    - Error categorization and severity
    - Component context information
    - Recovery guidance and suggestions
    - Timestamp and correlation tracking
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.BUSINESS_LOGIC,
        component: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
        recovery_suggestions: list | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.severity = severity
        self.category = category
        self.component = component
        self.context = context or {}
        self.recoverable = recoverable
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()
        self.error_id = f"err_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(message) % 10000}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "context": self.context,
            "recoverable": self.recoverable,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        """Enhanced string representation with context"""
        base = f"[{self.severity.value.upper()}] {self.message}"
        if self.component:
            base = f"[{self.component}] {base}"
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base += f" (Context: {context_str})"
        return base


class ConfigurationException(GPTTraderException):
    """Configuration-related errors"""

    def __init__(self, message: str, config_key: str | None = None, **kwargs: Any) -> None:
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key

        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check configuration file syntax and values",
                "Verify environment variables are set correctly",
                "Validate configuration against schema",
            ],
            **kwargs,
        )


class ValidationException(GPTTraderException):
    """Data validation errors"""

    def __init__(
        self, message: str, field: str | None = None, value: Any = None, **kwargs: Any
    ) -> None:
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["invalid_value"] = str(value)

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.VALIDATION,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check input data format and values",
                "Verify data meets validation criteria",
                "Review field constraints and limits",
            ],
            **kwargs,
        )


class TradingException(GPTTraderException):
    """Trading operation errors"""

    def __init__(
        self, message: str, order_id: str | None = None, symbol: str | None = None, **kwargs: Any
    ) -> None:
        context = kwargs.get("context", {})
        if order_id:
            context["order_id"] = order_id
        if symbol:
            context["symbol"] = symbol

        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.EXECUTION,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check order parameters and market conditions",
                "Verify account permissions and balance",
                "Review risk limits and position constraints",
            ],
            **kwargs,
        )


class RiskException(GPTTraderException):
    """Risk management errors"""

    def __init__(
        self,
        message: str,
        risk_type: str | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if risk_type:
            context["risk_type"] = risk_type
        if threshold is not None:
            context["threshold"] = threshold

        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=context,
            recoverable=False,  # Risk violations typically require manual intervention
            recovery_suggestions=[
                "Review risk parameters and limits",
                "Check portfolio exposure and concentration",
                "Verify risk calculation accuracy",
            ],
            **kwargs,
        )


class DataException(GPTTraderException):
    """Data quality and feed errors"""

    def __init__(
        self, message: str, data_source: str | None = None, symbol: str | None = None, **kwargs: Any
    ) -> None:
        context = kwargs.get("context", {})
        if data_source:
            context["data_source"] = data_source
        if symbol:
            context["symbol"] = symbol

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.EXTERNAL_API,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check data feed connection and status",
                "Verify API credentials and permissions",
                "Review data quality and completeness",
            ],
            **kwargs,
        )


class DatabaseException(GPTTraderException):
    """Database operation errors"""

    def __init__(
        self, message: str, operation: str | None = None, table: str | None = None, **kwargs: Any
    ) -> None:
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        if table:
            context["table"] = table

        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.DATABASE,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check database connection and permissions",
                "Verify table schema and constraints",
                "Review transaction isolation and locking",
            ],
            **kwargs,
        )


class NetworkException(GPTTraderException):
    """Network and connectivity errors"""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if endpoint:
            context["endpoint"] = endpoint
        if status_code:
            context["status_code"] = status_code

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.NETWORK,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check network connectivity and DNS resolution",
                "Verify API endpoints and authentication",
                "Review rate limiting and retry policies",
            ],
            **kwargs,
        )


class ComponentException(GPTTraderException):
    """Component lifecycle and integration errors"""

    def __init__(
        self,
        message: str,
        component_type: str | None = None,
        lifecycle_stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if component_type:
            context["component_type"] = component_type
        if lifecycle_stage:
            context["lifecycle_stage"] = lifecycle_stage

        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.EXECUTION,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check component configuration and dependencies",
                "Verify component initialization order",
                "Review resource availability and permissions",
            ],
            **kwargs,
        )


class ResourceException(GPTTraderException):
    """Resource availability and management errors"""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        current_usage: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if current_usage is not None:
            context["current_usage"] = current_usage

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.RESOURCE,
            context=context,
            recoverable=True,
            recovery_suggestions=[
                "Check system resource availability",
                "Review resource limits and quotas",
                "Consider scaling or optimization",
            ],
            **kwargs,
        )


# Convenience functions for common error patterns


def raise_config_error(
    message: str, config_key: str | None = None, component: str | None = None
) -> Never:
    """Raise a configuration error with standard context"""
    raise ConfigurationException(message, config_key=config_key, component=component)


def raise_validation_error(
    message: str, field: str | None = None, value: Any = None, component: str | None = None
) -> Never:
    """Raise a validation error with standard context"""
    raise ValidationException(message, field=field, value=value, component=component)


def raise_trading_error(
    message: str,
    order_id: str | None = None,
    symbol: str | None = None,
    component: str | None = None,
) -> Never:
    """Raise a trading error with standard context"""
    raise TradingException(message, order_id=order_id, symbol=symbol, component=component)


def raise_risk_error(
    message: str,
    risk_type: str | None = None,
    threshold: float | None = None,
    component: str | None = None,
) -> Never:
    """Raise a risk error with standard context"""
    raise RiskException(message, risk_type=risk_type, threshold=threshold, component=component)


def raise_data_error(
    message: str,
    data_source: str | None = None,
    symbol: str | None = None,
    component: str | None = None,
) -> Never:
    """Raise a data error with standard context"""
    raise DataException(message, data_source=data_source, symbol=symbol, component=component)
