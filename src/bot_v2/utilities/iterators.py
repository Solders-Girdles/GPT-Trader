"""Iterator utilities for stream processing and transport stubs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def empty_stream() -> Iterator[dict[str, Any]]:
    """Return an empty iterator for stub/noop transport streams.

    This is used in test stubs and no-op transports to provide a clean,
    type-safe empty stream instead of using the `if False: yield {}` pattern.

    Returns:
        Empty iterator that yields no items

    Example:
        >>> stream = empty_stream()
        >>> list(stream)
        []
        >>> from collections.abc import Iterator
        >>> isinstance(stream, Iterator)
        True
    """
    return iter(())


__all__ = ["empty_stream"]
