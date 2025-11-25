"""Error types and mappers for Coinbase adapter."""

from gpt_trader.features.brokerages.core.interfaces import (
    AuthError as CoreAuthError,
)
from gpt_trader.features.brokerages.core.interfaces import (
    BrokerageError as CoreBrokerageError,
)
from gpt_trader.features.brokerages.core.interfaces import (
    InsufficientFunds as CoreInsufficientFunds,
)
from gpt_trader.features.brokerages.core.interfaces import (
    InvalidRequestError as CoreInvalidRequestError,
)
from gpt_trader.features.brokerages.core.interfaces import (
    NotFoundError as CoreNotFoundError,
)
from gpt_trader.features.brokerages.core.interfaces import (
    PermissionDeniedError as CorePermissionDeniedError,
)
from gpt_trader.features.brokerages.core.interfaces import (
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
    if status == 403:
        return PermissionDeniedError(message or code or "permission denied")
    if "insufficient" in text:
        return InsufficientFunds(message or code or "insufficient funds")

    return BrokerageError(message or code or f"Status {status}")
