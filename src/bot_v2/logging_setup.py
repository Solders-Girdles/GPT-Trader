"""
Centralized logging setup for GPT-Trader V2.

Features:
- Rotating file handlers for general and critical logs
- Optional JSONL handlers for structured logs (used by ProductionLogger)
- Environment-driven debug mode
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from .system_paths import LOG_DIR, ensure_directories


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def configure_logging() -> None:
    """Configure rotating file logging and debug levels.

    - General text log: var/logs/perps_trading.log (INFO+)
    - Critical text log: var/logs/critical_events.log (WARNING+)
    - JSON logs (message-only):
        - var/logs/perps_trading.jsonl (DEBUG+)
        - var/logs/critical_events.jsonl (WARNING+)
    - Debug mode via PERPS_DEBUG=1
    - sizes configurable via env vars
    """
    ensure_directories((LOG_DIR,))
    log_dir = Path(os.getenv("PERPS_LOG_DIR", str(LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger and console
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    existing_targets = {
        getattr(h, "baseFilename", None) for h in root.handlers if hasattr(h, "baseFilename")
    }

    # Console handler (only add if not present)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(console)

    # File sizes
    general_max_bytes = int(os.getenv("PERPS_LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
    general_backups = int(os.getenv("PERPS_LOG_BACKUP_COUNT", "10"))
    critical_max_bytes = int(os.getenv("PERPS_CRIT_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
    critical_backups = int(os.getenv("PERPS_CRIT_LOG_BACKUP_COUNT", "5"))

    # General file handler (text)
    general_path = str(log_dir / "perps_trading.log")
    if general_path not in existing_targets:
        general_handler = logging.handlers.RotatingFileHandler(
            general_path, maxBytes=general_max_bytes, backupCount=general_backups
        )
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(general_handler)

    # Critical file handler (text)
    critical_path = str(log_dir / "critical_events.log")
    if critical_path not in existing_targets:
        critical_handler = logging.handlers.RotatingFileHandler(
            critical_path, maxBytes=critical_max_bytes, backupCount=critical_backups
        )
        critical_handler.setLevel(logging.WARNING)
        critical_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(critical_handler)

    # JSON loggers: dedicated logger that writes message-only JSON lines
    json_logger = logging.getLogger("bot_v2.json")
    json_logger.setLevel(logging.DEBUG)
    json_logger.propagate = False

    existing_json_targets = {
        getattr(h, "baseFilename", None) for h in json_logger.handlers if hasattr(h, "baseFilename")
    }
    json_formatter = logging.Formatter("%(message)s")  # message is pre-formatted JSON

    json_general_path = str(log_dir / "perps_trading.jsonl")
    if json_general_path not in existing_json_targets:
        json_general_handler = logging.handlers.RotatingFileHandler(
            json_general_path, maxBytes=general_max_bytes, backupCount=general_backups
        )
        json_general_handler.setLevel(logging.DEBUG)
        json_general_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_general_handler)

    json_critical_path = str(log_dir / "critical_events.jsonl")
    if json_critical_path not in existing_json_targets:
        json_critical_handler = logging.handlers.RotatingFileHandler(
            json_critical_path, maxBytes=critical_max_bytes, backupCount=critical_backups
        )
        json_critical_handler.setLevel(logging.WARNING)
        json_critical_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_critical_handler)

    # Debug mode controls
    if _env_flag("PERPS_DEBUG", "0"):
        # Raise root console/file logs to DEBUG for targeted packages
        logging.getLogger("bot_v2.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("bot_v2.orchestration").setLevel(logging.DEBUG)
