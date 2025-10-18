"""Spot profile loading and rule lookups for strategy orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="spot_profile_service")


def _load_spot_profile(path: Path) -> dict[str, Any]:
    """Load spot strategy profile YAML into a dictionary."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


class SpotProfileService:
    """Loads spot strategy profiles and returns per-symbol rule dictionaries."""

    def __init__(
        self,
        *,
        loader: Callable[[Path], dict[str, Any]] = _load_spot_profile,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self._loader = loader
        self._rules: dict[str, dict[str, Any]] = {}
        self._last_path: Path | None = None
        self._settings_locked = settings is not None
        self._settings = settings or load_runtime_settings()

    # ------------------------------------------------------------------
    def load(self, symbols: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not self._settings_locked:
            self._settings = load_runtime_settings()

        raw_env = self._settings.raw_env
        profile_path = Path(raw_env.get("SPOT_PROFILE_PATH", "config/profiles/spot.yaml"))
        if not profile_path.exists():
            logger.info(
                "Spot profile not found; using default parameters",
                operation="spot_profile_service",
                stage="load_missing",
                profile_path=str(profile_path),
            )
            self._rules = {}
            self._last_path = profile_path
            return {}
        try:
            profile_doc = self._loader(profile_path)
        except Exception as exc:
            logger.warning(
                "Failed to load spot profile",
                operation="spot_profile_service",
                stage="load_failure",
                profile_path=str(profile_path),
                error=str(exc),
                exc_info=True,
            )
            self._rules = {}
            self._last_path = profile_path
            return {}

        strategies = profile_doc.get("strategy", {}) if isinstance(profile_doc, dict) else {}
        resolved: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            rule = self._resolve_rule_for_symbol(symbol, strategies, profile_path)
            if rule is not None:
                resolved[symbol] = rule
        self._rules = resolved
        self._last_path = profile_path
        return dict(self._rules)

    def get(self, symbol: str) -> dict[str, Any]:
        return self._rules.get(symbol, {})

    def all_rules(self) -> dict[str, dict[str, Any]]:
        return dict(self._rules)

    @property
    def profile_path(self) -> Path | None:
        return self._last_path

    # ------------------------------------------------------------------
    def _resolve_rule_for_symbol(
        self, symbol: str, strategies: dict[str, Any], profile_path: Path
    ) -> dict[str, Any] | None:
        keys = [symbol, symbol.lower(), symbol.upper()]
        if "-" in symbol:
            base = symbol.split("-")[0]
            keys.extend([base, base.lower(), base.upper()])
        for key in keys:
            if key in strategies:
                return strategies.get(key) or {}
        logger.warning(
            "No strategy entry for symbol in profile; defaults will be used",
            operation="spot_profile_service",
            stage="missing_strategy",
            symbol=symbol,
            profile_path=str(profile_path),
        )
        return None
