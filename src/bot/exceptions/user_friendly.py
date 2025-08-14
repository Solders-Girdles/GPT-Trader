"""User-friendly error handling with recovery suggestions.

This module provides enhanced error messages that help users understand
what went wrong and how to fix it.
"""

import traceback
from datetime import datetime


class UserFriendlyError(Exception):
    """Base class for user-friendly errors with recovery suggestions."""

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        suggestions: list[str] | None = None,
        error_code: str | None = None,
    ):
        """Initialize user-friendly error.

        Args:
            message: The main error message
            cause: The underlying exception that caused this error
            suggestions: List of recovery suggestions
            error_code: Optional error code for reference
        """
        super().__init__(message)
        self.cause = cause
        self.suggestions = suggestions or []
        self.error_code = error_code

    def get_full_message(self) -> str:
        """Get the full error message with suggestions."""
        lines = [f"❌ Error: {str(self)}"]

        if self.error_code:
            lines.append(f"   Code: {self.error_code}")

        if self.cause:
            lines.append(f"   Caused by: {type(self.cause).__name__}: {str(self.cause)}")

        if self.suggestions:
            lines.append("\n💡 Suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"   {i}. {suggestion}")

        return "\n".join(lines)


class DataError(UserFriendlyError):
    """Error related to data issues."""

    def __init__(self, message: str, symbol: str | None = None, **kwargs):
        suggestions = [
            "Check if the symbol exists and is tradeable",
            "Verify the date range has market data available",
            "Try using a different data source",
            "Check your internet connection",
        ]

        if symbol:
            suggestions.insert(0, f"Verify '{symbol}' is a valid ticker symbol")

        super().__init__(message, suggestions=suggestions, error_code="DATA_ERROR", **kwargs)


class ConfigurationError(UserFriendlyError):
    """Error related to configuration issues."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        suggestions = [
            "Check your .env.local file for missing values",
            "Run 'gpt-trader wizard' for guided setup",
            "Verify all required environment variables are set",
            "Check the documentation for configuration examples",
        ]

        if config_key:
            suggestions.insert(0, f"Set the '{config_key}' configuration value")

        super().__init__(message, suggestions=suggestions, error_code="CONFIG_ERROR", **kwargs)


class APIError(UserFriendlyError):
    """Error related to API issues."""

    def __init__(self, message: str, api_name: str | None = None, **kwargs):
        suggestions = [
            "Check your API credentials are correct",
            "Verify your API subscription is active",
            "Check if you've exceeded rate limits",
            "Try again in a few moments",
            "Use DEMO_MODE=true for testing without API access",
        ]

        if api_name:
            suggestions.insert(0, f"Verify your {api_name} API key and secret")

        super().__init__(message, suggestions=suggestions, error_code="API_ERROR", **kwargs)


class StrategyError(UserFriendlyError):
    """Error related to strategy issues."""

    def __init__(self, message: str, strategy_name: str | None = None, **kwargs):
        suggestions = [
            "Check if the strategy is properly configured",
            "Verify strategy parameters are within valid ranges",
            "Try using a different strategy",
            "Check the strategy documentation for requirements",
        ]

        if strategy_name:
            suggestions.append(
                f"Run 'gpt-trader backtest --help' to see {strategy_name} parameters"
            )

        super().__init__(message, suggestions=suggestions, error_code="STRATEGY_ERROR", **kwargs)


class FileError(UserFriendlyError):
    """Error related to file operations."""

    def __init__(self, message: str, file_path: str | None = None, **kwargs):
        suggestions = [
            "Check if the file exists",
            "Verify you have read/write permissions",
            "Ensure the directory exists",
            "Check if the file is in use by another process",
        ]

        if file_path:
            suggestions.insert(0, f"Check if '{file_path}' is accessible")

        super().__init__(message, suggestions=suggestions, error_code="FILE_ERROR", **kwargs)


class ValidationError(UserFriendlyError):
    """Error related to input validation."""

    def __init__(self, message: str, field: str | None = None, **kwargs):
        suggestions = [
            "Check the input format matches requirements",
            "Verify all required fields are provided",
            "Use --help to see valid parameter formats",
            "Check the documentation for examples",
        ]

        if field:
            suggestions.insert(0, f"Verify the '{field}' value is valid")

        super().__init__(message, suggestions=suggestions, error_code="VALIDATION_ERROR", **kwargs)


class ErrorHandler:
    """Central error handler with user-friendly messages."""

    # Mapping of exception types to user-friendly error classes
    ERROR_MAPPINGS: dict[type[Exception], type[UserFriendlyError]] = {
        FileNotFoundError: FileError,
        PermissionError: FileError,
        ValueError: ValidationError,
        KeyError: ConfigurationError,
        ConnectionError: APIError,
        TimeoutError: APIError,
    }

    @classmethod
    def handle(cls, error: Exception, context: str | None = None) -> UserFriendlyError:
        """Convert an exception to a user-friendly error.

        Args:
            error: The exception to handle
            context: Optional context about what was happening

        Returns:
            A UserFriendlyError with helpful suggestions
        """
        # If already user-friendly, return as is
        if isinstance(error, UserFriendlyError):
            return error

        # Look for a mapping
        error_class = cls.ERROR_MAPPINGS.get(type(error), UserFriendlyError)

        # Create user-friendly version
        message = str(error)
        if context:
            message = f"{context}: {message}"

        return error_class(message, cause=error)

    @classmethod
    def format_error(cls, error: Exception, verbose: bool = False) -> str:
        """Format an error for display.

        Args:
            error: The error to format
            verbose: Whether to include stack trace

        Returns:
            Formatted error string
        """
        # Convert to user-friendly if needed
        friendly_error = cls.handle(error)

        # Get the message
        message = friendly_error.get_full_message()

        # Add stack trace if verbose
        if verbose and friendly_error.cause:
            message += "\n\n📋 Stack Trace:\n"
            message += "".join(
                traceback.format_exception(
                    type(friendly_error.cause),
                    friendly_error.cause,
                    friendly_error.cause.__traceback__,
                )
            )

        return message


def handle_cli_error(func):
    """Decorator for CLI functions to provide user-friendly error handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert to user-friendly error
            friendly_error = ErrorHandler.handle(e, context=f"Running {func.__name__}")

            # Print the error
            print(friendly_error.get_full_message())

            # Re-raise for proper exit code
            raise SystemExit(1) from e

    return wrapper


# Common error scenarios with specific messages
class CommonErrors:
    """Common error scenarios with pre-configured messages."""

    @staticmethod
    def no_data(symbol: str, start: datetime, end: datetime) -> DataError:
        """Create error for no data available."""
        return DataError(
            f"No data available for {symbol} from {start.date()} to {end.date()}",
            symbol=symbol,
            suggestions=[
                f"Check if {symbol} was listed during this period",
                "Try a different date range",
                "Verify the symbol is correct (use uppercase)",
                "Check if markets were open during this period",
            ],
        )

    @staticmethod
    def invalid_date_range() -> ValidationError:
        """Create error for invalid date range."""
        return ValidationError(
            "Invalid date range: start date must be before end date",
            field="date range",
            suggestions=[
                "Ensure --start is before --end",
                "Use format YYYY-MM-DD for dates",
                "Example: --start 2024-01-01 --end 2024-12-31",
            ],
        )

    @staticmethod
    def missing_api_keys() -> ConfigurationError:
        """Create error for missing API keys."""
        return ConfigurationError(
            "API credentials not found",
            suggestions=[
                "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env.local",
                "Or use DEMO_MODE=true for testing without API keys",
                "Sign up for free at https://alpaca.markets",
                "Run 'gpt-trader wizard' for guided setup",
            ],
        )

    @staticmethod
    def strategy_not_found(strategy_name: str) -> StrategyError:
        """Create error for strategy not found."""
        return StrategyError(
            f"Strategy '{strategy_name}' not found",
            strategy_name=strategy_name,
            suggestions=[
                "Available strategies: demo_ma, trend_breakout",
                "Check spelling and case",
                "Use 'gpt-trader backtest --help' to see available strategies",
            ],
        )

    @staticmethod
    def insufficient_capital(required: float, available: float) -> ValidationError:
        """Create error for insufficient capital."""
        return ValidationError(
            f"Insufficient capital: ${available:,.2f} available, ${required:,.2f} required",
            field="capital",
            suggestions=[
                "Reduce the number of positions (--max-positions)",
                "Decrease position size (--risk-pct)",
                "Increase starting capital in configuration",
            ],
        )
