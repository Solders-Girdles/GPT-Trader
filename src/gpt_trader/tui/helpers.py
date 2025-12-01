"""
TUI Utilities.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

F = TypeVar("F", bound=Callable[..., Any])


def safe_update(func: F) -> F:
    """
    Decorator to wrap widget update methods.
    Catches exceptions, logs them, and prevents the app from crashing.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Try to get class name if it's a method
            context = func.__name__
            if args and hasattr(args[0], "__class__"):
                context = f"{args[0].__class__.__name__}.{func.__name__}"

            logger.error(f"Error in TUI update ({context}): {e}", exc_info=True)
            return None

    return wrapper  # type: ignore
