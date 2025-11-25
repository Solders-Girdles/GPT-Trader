"""Symbol processing compatibility module."""

from collections.abc import Callable
from typing import Any, Protocol


class SymbolProcessor(Protocol):
    def __call__(self, symbol: str) -> Any: ...


# Type alias for compatibility
_CallableSymbolProcessor = Callable[[str], Any]

__all__ = ["_CallableSymbolProcessor"]
