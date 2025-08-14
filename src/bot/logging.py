from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _ensure_root_config() -> None:
    """
    Configure a single stdout handler on the root logger exactly once.

    We do not 'force=True' by default to avoid clobbering pytest or user configs.
    If you ever see silence again due to an upstream logger stealing handlers,
    set LOG_FORCE=1 to override.
    """
    root = logging.getLogger()

    # Check if we already have a handler configured (by CLI or elsewhere)
    # This prevents duplicate handlers
    if root.handlers and os.getenv("LOG_FORCE", "0") != "1":
        return

    # Clear existing handlers if forcing.
    if root.handlers and os.getenv("LOG_FORCE", "0") == "1":
        for h in list(root.handlers):
            root.removeHandler(h)

    # Use centralized config if available; fall back to env
    try:
        from bot.config import get_config  # local import to avoid cycles

        level_name = getattr(get_config().logging, "level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
    except Exception:
        # Fallback to environment variable for bootstrap scenarios
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    # Only add handler if none exist
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(handler)

    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    _ensure_root_config()
    logger = logging.getLogger(name)

    # Use propagation to root logger, don't add individual handlers
    # This prevents duplicate logging
    logger.propagate = True

    # Don't add handlers to child loggers - let them propagate to root
    # This is the key fix for duplicate logging

    return logger
