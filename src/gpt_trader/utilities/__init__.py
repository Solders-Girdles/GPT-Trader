"""
Simplified Utilities.
"""
from .importing import optional_import
from .logging_patterns import get_logger, log_operation
from typing import AsyncIterator

async def empty_stream() -> AsyncIterator[None]:
    if False:
        yield None

__all__ = ["get_logger", "log_operation", "optional_import", "empty_stream"]
