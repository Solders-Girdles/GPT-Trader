"""Symbol processing compatibility module."""

from typing import Any, Callable, Protocol

class SymbolProcessor(Protocol):
    def __call__(self, symbol: str) -> Any: ...

# Type alias for compatibility
_CallableSymbolProcessor = Callable[[str], Any]

__all__ = ["_CallableSymbolProcessor"]
