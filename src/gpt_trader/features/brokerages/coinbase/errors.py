"""Error types and mappers for Coinbase adapter."""

from gpt_trader.core import (
    AuthError as CoreAuthError,
)
from gpt_trader.core import (
    BrokerageError as CoreBrokerageError,
)
from gpt_trader.core import (
    InsufficientFunds as CoreInsufficientFunds,
)
from gpt_trader.core import (
    InvalidRequestError as CoreInvalidRequestError,
)
from gpt_trader.core import (
    NotFoundError as CoreNotFoundError,
)
from gpt_trader.core import (
    PermissionDeniedError as CorePermissionDeniedError,
)
from gpt_trader.core import (
    RateLimitError as CoreRateLimitError,
)


class BrokerageError(CoreBrokerageError):
    """Base error for brokerage adapters."""


class RateLimitError(CoreRateLimitError, BrokerageError):
    pass


class AuthError(CoreAuthError, BrokerageError):
    pass


class NotFoundError(CoreNotFoundError, BrokerageError):
    pass


class InvalidRequestError(CoreInvalidRequestError, BrokerageError):
    pass


class InsufficientFunds(CoreInsufficientFunds, BrokerageError):
    pass


class PermissionDeniedError(CorePermissionDeniedError, BrokerageError):
    pass


class TransientBrokerError(BrokerageError):
    """Retryable error (network timeout, temporary unavailability).

    Callers should implement retry logic with backoff for this error type.
    """

    pass


class OrderCancellationError(BrokerageError):
    """Order cancellation failed.

    Attributes:
        order_id: The ID of the order that failed to cancel.
    """

    def __init__(self, message: str, order_id: str | None = None) -> None:
        super().__init__(message)
        self.order_id = order_id


class OrderQueryError(BrokerageError):
    """Failed to query order information."""

    pass


def map_http_error(status: int, code: str | None, message: str | None) -> BrokerageError:
    text = (message or code or "").lower()
    if status in (401, 407) or (
        code and code.lower() in {"invalid_api_key", "invalid_signature", "authentication_error"}
    ):
        return AuthError(message or code or "authentication failed")
    if status == 404:
        return NotFoundError(message or "not found")
    if status == 429:
        return RateLimitError(message or code or "rate limited")
    if status == 503 or (
        code and code.lower() in {"service_unavailable", "temporarily_unavailable"}
    ):
        return TransientBrokerError(message or code or "service unavailable")
    if status == 403:
        return PermissionDeniedError(message or code or "permission denied")
    if "insufficient" in text:
        return InsufficientFunds(message or code or "insufficient funds")

    return BrokerageError(message or code or f"Status {status}")
