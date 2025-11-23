"""Error types and mappers for Coinbase adapter."""

class BrokerageError(Exception):
    """Base error for brokerage adapters."""

class RateLimitError(BrokerageError):
    pass

class AuthError(BrokerageError):
    pass

class NotFoundError(BrokerageError):
    pass

class InvalidRequestError(BrokerageError):
    pass

class InsufficientFunds(BrokerageError):
    pass

class PermissionDeniedError(BrokerageError):
    pass

def map_http_error(status: int, code: str | None, message: str | None) -> BrokerageError:
    text = (message or code or "").lower()
    if status in (401, 407) or (code and code.lower() in {"invalid_api_key", "invalid_signature", "authentication_error"}):
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
