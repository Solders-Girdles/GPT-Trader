"""Pre-trade validation exceptions."""


class ValidationError(Exception):
    """Risk validation failure with clear message."""


__all__ = ["ValidationError"]
