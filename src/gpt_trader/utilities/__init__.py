"""
Simplified Utilities.
"""

from collections.abc import AsyncIterator

from .datetime_helpers import utc_now
from .importing import optional_import
from .logging_patterns import get_logger, log_operation


async def empty_stream() -> AsyncIterator[None]:
    if False:
        yield None


__all__ = ["get_logger", "log_operation", "optional_import", "empty_stream", "utc_now"]
