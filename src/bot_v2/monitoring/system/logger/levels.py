"""Logging level definitions for production logger."""

from __future__ import annotations

import logging
from enum import Enum


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


__all__ = ["LogLevel", "LEVEL_MAP"]
