"""
Error classes for bot_v2.
"""


class BotError(Exception):
    """Base error for bot operations."""
    pass


class ExecutionError(BotError):
    """Error during order execution."""
    pass


class ValidationError(BotError):
    """Validation error."""
    pass


class ConfigError(BotError):
    """Configuration error."""
    pass


class RiskError(BotError):
    """Risk management error."""
    pass