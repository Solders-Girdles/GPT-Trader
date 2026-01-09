"""Centralized logging setup for GPT-Trader V2."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

from gpt_trader.config.path_registry import LOG_DIR, ensure_directories
from gpt_trader.logging.json_formatter import (
    StructuredJSONFormatterWithTimestamp,
)


def _env_flag(
    name: str,
    default: str = "0",
) -> bool:
    raw_value = os.environ.get(name, default)
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _env_lookup(*keys: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value for the given keys."""

    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def configure_logging(config: Any = None, tui_mode: bool = False) -> None:
    """
    Configure rotating file logging and debug levels.

    Args:
        config: Optional configuration object (unused, kept for backward compatibility)
        tui_mode: If True, suppress console StreamHandler to avoid corrupting TUI display.
                  Logs will still be written to files and can be displayed via TuiLogHandler.
    """

    ensure_directories((LOG_DIR,))
    log_dir_raw = _env_lookup("COINBASE_TRADER_LOG_DIR", "PERPS_LOG_DIR")
    log_dir = Path(log_dir_raw) if log_dir_raw else Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    existing_targets = {
        getattr(handler, "baseFilename", None)
        for handler in root.handlers
        if hasattr(handler, "baseFilename")
    }

    # Only add StreamHandler if NOT in TUI mode (prevents corrupting Textual display)
    if not tui_mode:
        # Check for console StreamHandlers (exclude file handlers and test fixtures)
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and type(h).__name__ not in {"LogCaptureHandler", "LogCaptureFixture"}
        ]
        if not console_handlers:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            root.addHandler(console)

    general_max_bytes = int(
        _env_lookup("COINBASE_TRADER_LOG_MAX_BYTES", "PERPS_LOG_MAX_BYTES") or str(50 * 1024 * 1024)
    )
    general_backups = int(
        _env_lookup("COINBASE_TRADER_LOG_BACKUP_COUNT", "PERPS_LOG_BACKUP_COUNT") or "10"
    )
    critical_max_bytes = int(
        _env_lookup("COINBASE_TRADER_CRIT_LOG_MAX_BYTES", "PERPS_CRIT_LOG_MAX_BYTES")
        or str(10 * 1024 * 1024)
    )
    critical_backups = int(
        _env_lookup("COINBASE_TRADER_CRIT_LOG_BACKUP_COUNT", "PERPS_CRIT_LOG_BACKUP_COUNT") or "5"
    )

    general_path = str(log_dir / "coinbase_trader.log")
    if general_path not in existing_targets:
        general_handler = logging.handlers.RotatingFileHandler(
            general_path,
            maxBytes=general_max_bytes,
            backupCount=general_backups,
        )
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(general_handler)

    critical_path = str(log_dir / "critical_events.log")
    if critical_path not in existing_targets:
        critical_handler = logging.handlers.RotatingFileHandler(
            critical_path,
            maxBytes=critical_max_bytes,
            backupCount=critical_backups,
        )
        critical_handler.setLevel(logging.WARNING)
        critical_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(critical_handler)

    json_logger = logging.getLogger("gpt_trader.json")
    json_logger.setLevel(logging.DEBUG)
    json_logger.propagate = False

    existing_json_targets = {
        getattr(handler, "baseFilename", None)
        for handler in json_logger.handlers
        if hasattr(handler, "baseFilename")
    }
    # Use the enhanced JSON formatter with correlation ID and domain field support
    json_formatter = StructuredJSONFormatterWithTimestamp(
        ensure_ascii=False,
        sort_keys=True,
    )

    json_general_path = str(log_dir / "coinbase_trader.jsonl")
    if json_general_path not in existing_json_targets:
        json_general_handler = logging.handlers.RotatingFileHandler(
            json_general_path,
            maxBytes=general_max_bytes,
            backupCount=general_backups,
        )
        json_general_handler.setLevel(logging.DEBUG)
        json_general_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_general_handler)

    json_critical_path = str(log_dir / "critical_events.jsonl")
    if json_critical_path not in existing_json_targets:
        json_critical_handler = logging.handlers.RotatingFileHandler(
            json_critical_path,
            maxBytes=critical_max_bytes,
            backupCount=critical_backups,
        )
        json_critical_handler.setLevel(logging.WARNING)
        json_critical_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_critical_handler)

    if _env_flag("COINBASE_TRADER_DEBUG", "0") or _env_flag("PERPS_DEBUG", "0"):
        logging.getLogger("gpt_trader.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("gpt_trader.features.live_trade").setLevel(logging.DEBUG)

    # Attach TUI handler early when in TUI mode to capture startup logs
    if tui_mode:
        try:
            from gpt_trader.tui.log_manager import attach_tui_log_handler

            attach_tui_log_handler()
        except ImportError:
            # TUI dependencies not installed, skip handler attachment
            pass
