"""Tests for CLI config building helpers."""

from __future__ import annotations

from argparse import Namespace

import pytest

from gpt_trader.app.config.profile_loader import DEFAULT_RUNTIME_PROFILE_NAME
from gpt_trader.cli import services
from gpt_trader.config.types import Profile


def _make_args(**overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "config": None,
        "dry_run": False,
        "symbols": None,
        "interval": None,
        "target_leverage": None,
        "reduce_only_mode": False,
        "time_in_force": None,
        "enable_order_preview": False,
        "account_telemetry_interval": None,
    }
    defaults.setdefault("profile", DEFAULT_RUNTIME_PROFILE_NAME)
    defaults.update(overrides)
    return Namespace(**defaults)  # type: ignore[arg-type]


def test_default_profile_applied_when_missing() -> None:
    args = _make_args()
    delattr(args, "profile")

    config = services.build_config_from_args(args)

    assert config.profile == Profile.DEV
    assert config.profile.value == DEFAULT_RUNTIME_PROFILE_NAME


def test_profile_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_SYMBOLS", "BTC-PERP,ETH-PERP")
    monkeypatch.setenv("INTERVAL", "5")

    args = _make_args(profile="dev")
    config = services.build_config_from_args(args)

    assert config.symbols == ["BTC-USD", "ETH-USD"]
    assert config.interval == 60


def test_cli_args_override_profile_settings() -> None:
    args = _make_args(profile="dev", symbols=["DOGE-PERP"], interval=10)

    config = services.build_config_from_args(args)

    assert config.symbols == ["DOGE-PERP"]
    assert config.interval == 10
