from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging API hook
        self.records.append(record)


@contextmanager
def captured_logger(name: str, level: int = logging.DEBUG) -> tuple[logging.Logger, ListHandler]:
    logger = logging.getLogger(name)
    handler = ListHandler()
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield logger, handler
    finally:
        logger.handlers = []


def messages(handler: ListHandler) -> list[str]:
    return [record.getMessage() for record in handler.records]


def has_attr(record: logging.LogRecord, attr: str, value: Any = None) -> bool:
    """Check if a log record has an attribute, optionally with a specific value."""
    if not hasattr(record, attr):
        return False
    if value is None:
        return True
    return getattr(record, attr) == value
