"""Error types and mappers for Coinbase adapter (scaffold)."""

from __future__ import annotations

from ..core.interfaces import (
    AuthError,
    BrokerageError,
    InsufficientFunds,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)


def map_http_error(status: int, code: str | None, message: str | None) -> BrokerageError:
    text = (message or code or "").lower()
    # Auth
    if status in (401, 407) or (
        code and code.lower() in {"invalid_api_key", "invalid_signature", "authentication_error"}
    ):
        return AuthError(message or code or "authentication failed")
    # Not found
    if status == 404:
        return NotFoundError(message or "not found")
    # Rate limit
    if status == 429 or (code and "rate" in code.lower()) or "rate limit" in text:
        return RateLimitError(message or code or "rate limited")
    # Permission denied
    if status == 403 or "permission" in text or "forbidden" in text:
        return PermissionDeniedError(message or code or "permission denied")
    # Insufficient funds
    if (code and "insufficient" in code.lower()) or "insufficient funds" in text:
        return InsufficientFunds(message or code or "insufficient funds")
    # Duplicate client order id (idempotency)
    if ("duplicate" in text and "client" in text) or "client_order_id" in text and "dup" in text:
        return InvalidRequestError(message or code or "duplicate client_order_id")
    # Specific invalid request variants for clearer diagnostics
    if "post only" in text and "cross" in text:
        return InvalidRequestError("post_only_would_cross")
    if "reduce only" in text:
        return InvalidRequestError("reduce_only_violation")
    if "min size" in text or "minimum size" in text:
        return InvalidRequestError("min_size_violation")
    if "max size" in text or "maximum size" in text:
        return InvalidRequestError("max_size_violation")
    if "leverage" in text and ("exceed" in text or "invalid" in text):
        return InvalidRequestError("leverage_violation")
    if "invalid" in text and ("price" in text or "size" in text):
        return InvalidRequestError("invalid_price_or_size")
    # Generic invalid request / size / price
    if status == 400 or any(
        k in text for k in ["invalid", "size", "price", "post only", "reduce only", "leverage"]
    ):
        return InvalidRequestError(message or code or "invalid_request")
    # Fallback
    return BrokerageError(message or code or "unknown error")
