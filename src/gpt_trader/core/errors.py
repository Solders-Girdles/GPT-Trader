"""Core exceptions used across all slices."""


class InvalidRequestError(Exception):
    """Invalid request parameters."""

    pass


class InsufficientFunds(Exception):
    """Insufficient funds for operation."""

    pass


class NotFoundError(Exception):
    """Resource not found."""

    pass


class AuthError(Exception):
    """Authentication or authorization error."""

    pass


class BrokerageError(Exception):
    """General brokerage/exchange error."""

    pass


class RateLimitError(Exception):
    """API rate limit exceeded."""

    pass


class PermissionDeniedError(Exception):
    """Permission denied for operation."""

    pass
