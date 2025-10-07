"""Helper utilities for CLI command implementations."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Iterable

from bot_v2.orchestration.bootstrap import build_bot
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot import PerpsBot

DEFAULT_SKIP_KEYS = {"profile", "command", "handler"}
ALIAS_KEYS = {"interval": "update_interval"}


def build_config_from_args(
    args: Namespace,
    *,
    include: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
) -> BotConfig:
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

    if "symbols" not in overrides or not overrides.get("symbols"):
        env_symbols = os.getenv("TRADING_SYMBOLS", "")
        if env_symbols:
            tokens = [
                tok.strip() for tok in env_symbols.replace(";", ",").split(",") if tok.strip()
            ]
            if tokens:
                overrides["symbols"] = tokens

    profile = getattr(args, "profile", "dev")
    return BotConfig.from_profile(profile, **overrides)


def instantiate_bot(config: BotConfig) -> PerpsBot:
    bot, _registry = build_bot(config)
    return bot
