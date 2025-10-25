"""Helper utilities for CLI command implementations."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable

from bot_v2.orchestration.bootstrap import build_bot
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.runtime_settings import (
    DEFAULT_RUNTIME_SETTINGS_PROVIDER,
    RuntimeSettings,
    RuntimeSettingsProvider,
)

DEFAULT_SKIP_KEYS = {"profile", "command", "handler"}
ALIAS_KEYS = {"interval": "update_interval"}

OVERRIDE_SETTINGS: RuntimeSettings | None = None
SETTINGS_PROVIDER: RuntimeSettingsProvider = DEFAULT_RUNTIME_SETTINGS_PROVIDER


def _resolve_settings(settings: RuntimeSettings | None) -> RuntimeSettings:
    if settings is not None:
        return settings
    if OVERRIDE_SETTINGS is not None:
        return OVERRIDE_SETTINGS
    return SETTINGS_PROVIDER.get()


def build_config_from_args(
    args: Namespace,
    *,
    include: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
    settings: RuntimeSettings | None = None,
) -> BotConfig:
    runtime_settings = _resolve_settings(settings)

    skip_set = set(DEFAULT_SKIP_KEYS)
    if skip:
        skip_set.update(skip)

    include_set = set(include) if include is not None else None
    overrides: dict[str, object] = {}

    for key, value in vars(args).items():
        if key in skip_set or value is None:
            continue
        if include_set is not None and key not in include_set:
            continue
        overrides[key] = value

    for alias, target in ALIAS_KEYS.items():
        if alias in overrides and target not in overrides:
            overrides[target] = overrides.pop(alias)

    allow_env_symbols = not overrides.get("symbols")
    if allow_env_symbols:
        env_symbols = runtime_settings.raw_env.get("TRADING_SYMBOLS", "")
        if env_symbols:
            tokens = [
                tok.strip() for tok in env_symbols.replace(";", ",").split(",") if tok.strip()
            ]
            if tokens:
                overrides["symbols"] = tokens

    profile = getattr(args, "profile", "dev")
    return BotConfig.from_profile(profile, settings=runtime_settings, **overrides)


def instantiate_bot(config: BotConfig) -> PerpsBot:
    """Instantiate a PerpsBot using the modern ApplicationContainer approach."""
    bot, _ = build_bot(config, settings_provider=SETTINGS_PROVIDER)
    return bot
