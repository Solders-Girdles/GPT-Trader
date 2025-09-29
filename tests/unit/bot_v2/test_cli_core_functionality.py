"""Core CLI behavior tests using real subprocess invocations."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.unit.bot_v2.cli_test_utils import cli_env


def test_cli_help_output():
    """CLI should display help text with key options."""
    result = subprocess.run(
        [sys.executable, "-m", "bot_v2.cli", "--help"],
        capture_output=True,
        text=True,
        env=cli_env(),
    )

    assert result.returncode == 0
    stdout = result.stdout
    assert "Perpetuals Trading Bot" in stdout
    assert "--profile" in stdout
    assert "--dry-run" in stdout
    assert "--symbols" in stdout
    assert "--reduce-only" in stdout


def test_cli_invalid_profile_rejected():
    """Passing an unsupported profile should fail fast."""
    result = subprocess.run(
        [sys.executable, "-m", "bot_v2.cli", "--profile", "invalid_profile"],
        capture_output=True,
        text=True,
        env=cli_env(),
    )

    assert result.returncode != 0
    assert "invalid choice: 'invalid_profile'" in result.stderr


def test_cli_argument_propagation(tmp_path):
    """CLI-style arguments should flow through to the generated BotConfig."""
    dump_script = tmp_path / "dump_config.py"
    dump_script.write_text(
        """
import sys
import json
from bot_v2.orchestration.configuration import BotConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="dev")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--symbols", nargs="+")
parser.add_argument("--interval", type=int)
parser.add_argument("--leverage", dest="target_leverage", type=int)
parser.add_argument("--reduce-only", dest="reduce_only_mode", action="store_true")
parser.add_argument("--dev-fast", action="store_true")
args = parser.parse_args()

config_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "profile"}
config = BotConfig.from_profile(args.profile, **config_overrides)

print(json.dumps({
    "symbols": config.symbols,
    "update_interval": getattr(config, "update_interval", getattr(config, "interval", None)),
    "target_leverage": getattr(config, "target_leverage", None),
    "reduce_only_mode": config.reduce_only_mode,
    "dry_run": config.dry_run,
}))
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            str(dump_script),
            "--profile",
            "dev",
            "--dry-run",
            "--symbols",
            "BTC-PERP",
            "ETH-PERP",
            "--interval",
            "5",
            "--leverage",
            "3",
            "--reduce-only",
        ],
        capture_output=True,
        text=True,
        env=cli_env(),
    )

    assert result.returncode == 0
    config = json.loads(result.stdout)
    assert config["symbols"] == ["BTC-PERP", "ETH-PERP"]
    assert config["update_interval"] == 5
    assert config["target_leverage"] is not None
    assert config["reduce_only_mode"] is True
    assert config["dry_run"] is True
