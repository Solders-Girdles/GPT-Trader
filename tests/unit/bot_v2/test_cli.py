"""CLI tests for bot_v2.cli.

Covers argument parsing and integration points without running the full bot.
"""

import builtins
import types
from types import SimpleNamespace

import asyncio
import pytest


def test_cli_dev_fast_runs_with_overrides(monkeypatch):
    # Arrange: fake argv
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "0")
    argv = [
        "prog",
        "--profile",
        "dev",
        "--dry-run",
        "--symbols",
        "BTC-PERP",
        "ETH-PERP",
        "--interval",
        "3",
        "--leverage",
        "2",
        "--reduce-only",
        "--dev-fast",
    ]

    # Dummy BotConfig.from_profile that returns a minimal object with requested attrs
    created_configs = {}

    class DummyConfig:
        def __init__(self, profile, **kw):
            self.profile = SimpleNamespace(value=profile)
            for k, v in kw.items():
                setattr(self, k, v)

    def fake_from_profile(profile, **overrides):
        created_configs["profile"] = profile
        created_configs["overrides"] = overrides
        return DummyConfig(profile, **overrides)

    ran = {"called": False, "single_cycle": None}

    class DummyBot:
        def __init__(self, config):
            # validate a couple of expected overrides propagated
            assert getattr(config, "dry_run", False) is True
            # CLI uses dest="reduce_only_mode" for the --reduce-only flag
            assert getattr(config, "reduce_only_mode", False) is True
            # Provide `running` attribute used by signal handler
            self.running = True

        async def run(self, *, single_cycle: bool = False):
            ran["called"] = True
            ran["single_cycle"] = single_cycle

    def fake_build_bot(config):
        dummy = DummyBot(config)
        return dummy, SimpleNamespace()

    # Patch imports inside cli module
    import importlib

    cli = importlib.import_module("bot_v2.cli")
    monkeypatch.setattr(cli, "BotConfig", SimpleNamespace(from_profile=fake_from_profile))
    monkeypatch.setattr(cli, "build_bot", fake_build_bot)

    # Patch sys.argv for argparse
    monkeypatch.setattr("sys.argv", argv)

    # Act
    exit_code = cli.main()

    # Assert
    assert exit_code == 0
    assert ran["called"] is True
    assert ran["single_cycle"] is True  # --dev-fast propagates to single_cycle
    # Ensure symbols and interval made it through overrides
    assert created_configs["profile"] == "dev"
    assert created_configs["overrides"]["symbols"] == ["BTC-PERP", "ETH-PERP"]
    assert created_configs["overrides"]["interval"] == 3
