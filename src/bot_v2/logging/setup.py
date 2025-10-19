"""Centralized logging setup for GPT-Trader V2."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from bot_v2.config.path_registry import LOG_DIR, ensure_directories
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings


def _env_flag(
    name: str,
    default: str = "0",
    *,
    settings: RuntimeSettings | None = None,
) -> bool:
    runtime_settings = settings or load_runtime_settings()
    raw_value = runtime_settings.raw_env.get(name, default)
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _env_lookup(raw_env: dict[str, str], *keys: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value for the given keys."""

    for key in keys:
        value = raw_env.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def configure_logging(settings: RuntimeSettings | None = None) -> None:
    """Configure rotating file logging and debug levels."""

    runtime_settings = settings or load_runtime_settings()
    raw_env = runtime_settings.raw_env

    ensure_directories((LOG_DIR,))
    log_dir_raw = _env_lookup(raw_env, "COINBASE_TRADER_LOG_DIR", "PERPS_LOG_DIR")
    log_dir = Path(log_dir_raw) if log_dir_raw else Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    existing_targets = {
        getattr(handler, "baseFilename", None)
        for handler in root.handlers
        if hasattr(handler, "baseFilename")
    }

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(console)

    general_max_bytes = int(
        _env_lookup(raw_env, "COINBASE_TRADER_LOG_MAX_BYTES", "PERPS_LOG_MAX_BYTES")
        or str(50 * 1024 * 1024)
    )
    general_backups = int(
        _env_lookup(raw_env, "COINBASE_TRADER_LOG_BACKUP_COUNT", "PERPS_LOG_BACKUP_COUNT") or "10"
    )
    critical_max_bytes = int(
        _env_lookup(raw_env, "COINBASE_TRADER_CRIT_LOG_MAX_BYTES", "PERPS_CRIT_LOG_MAX_BYTES")
        or str(10 * 1024 * 1024)
    )
    critical_backups = int(
        _env_lookup(raw_env, "COINBASE_TRADER_CRIT_LOG_BACKUP_COUNT", "PERPS_CRIT_LOG_BACKUP_COUNT")
        or "5"
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

    legacy_general_path = str(log_dir / "perps_trading.log")
    if legacy_general_path != general_path and legacy_general_path not in existing_targets:
        legacy_general_handler = logging.handlers.RotatingFileHandler(
            legacy_general_path,
            maxBytes=general_max_bytes,
            backupCount=general_backups,
        )
        legacy_general_handler.setLevel(logging.INFO)
        legacy_general_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(legacy_general_handler)

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

    json_logger = logging.getLogger("bot_v2.json")
    json_logger.setLevel(logging.DEBUG)
    json_logger.propagate = False

    existing_json_targets = {
        getattr(handler, "baseFilename", None)
        for handler in json_logger.handlers
        if hasattr(handler, "baseFilename")
    }
    json_formatter = logging.Formatter("%(message)s")

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

    legacy_json_general_path = str(log_dir / "perps_trading.jsonl")
    if (
        legacy_json_general_path != json_general_path
        and legacy_json_general_path not in existing_json_targets
    ):
        legacy_json_general_handler = logging.handlers.RotatingFileHandler(
            legacy_json_general_path,
            maxBytes=general_max_bytes,
            backupCount=general_backups,
        )
        legacy_json_general_handler.setLevel(logging.DEBUG)
        legacy_json_general_handler.setFormatter(json_formatter)
        json_logger.addHandler(legacy_json_general_handler)

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

    if _env_flag("COINBASE_TRADER_DEBUG", "0", settings=runtime_settings) or _env_flag(
        "PERPS_DEBUG", "0", settings=runtime_settings
    ):
        logging.getLogger("bot_v2.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("bot_v2.orchestration").setLevel(logging.DEBUG)
