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
    if root.handlers and os.getenv("LOG_FORCE", "0") != "1":
        return

    # Clear existing handlers if forcing.
    if root.handlers and os.getenv("LOG_FORCE", "0") == "1":
        for h in list(root.handlers):
            root.removeHandler(h)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))

    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    _ensure_root_config()
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
