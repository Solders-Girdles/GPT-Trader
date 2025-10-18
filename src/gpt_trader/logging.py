"""Centralised logging configuration for the ``gpt_trader`` package."""

from __future__ import annotations

import logging
from typing import Literal

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int = "INFO",
) -> None:
    """Initialise standard logging with a consistent formatter.

    Keeping this separate avoids circular imports when ``app`` wiring grows.
    """

    if isinstance(level, str):
        logging_level = logging.getLevelName(level.upper())
    else:
        logging_level = level

    logging.basicConfig(level=logging_level, format=DEFAULT_FORMAT)


__all__ = ["configure"]
