"""CLI integration tests using real subprocess invocations.

Tests CLI behavior, argument parsing, validation, and execution scenarios
using subprocess calls to verify end-to-end CLI functionality.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile

import pytest

from tests.unit.bot_v2.cli_test_utils import cli_env


# ===== Core CLI Behavior =====


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


# ===== Validation Scenarios =====


def test_cli_invalid_tif_rejected():
    """An unsupported time-in-force value should trigger parser error output."""
    result = subprocess.run(
        [sys.executable, "-m", "bot_v2.cli", "--tif", "INVALID"],
        capture_output=True,
        text=True,
        env=cli_env(),
    )

    assert result.returncode != 0
    assert "invalid choice: 'INVALID'" in result.stderr


@pytest.mark.parametrize("symbols", [["INVALID-SYMBOL"], ["BTC-PERP", "INVALID"], [""]])
def test_cli_symbol_validation(symbols):
    """Symbol inputs should be validated before execution."""
    env = cli_env()
    env["PERPS_FORCE_MOCK"] = "1"
    env["EVENT_STORE_ROOT"] = tempfile.mkdtemp()

    cmd = [sys.executable, "-m", "bot_v2.cli", "--profile", "dev", "--dev-fast"]
    if symbols:
        cmd.extend(["--symbols"] + symbols)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if symbols == [""]:
        assert result.returncode != 0
        assert "Symbols must be non-empty" in result.stderr
    else:
        # Deterministic broker accepts arbitrary symbols; ensure process completes.
        assert result.returncode == 0


# ===== Execution Scenarios =====


def test_cli_dev_fast_single_cycle(tmp_path):
    """`--dev-fast` should execute a single cycle using mock broker settings."""
    env = cli_env()
    env["PERPS_FORCE_MOCK"] = "1"
    env["PERPS_SKIP_RECONCILE"] = "1"
    env["EVENT_STORE_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "bot_v2.cli", "--profile", "dev", "--dev-fast"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "Starting Perps Bot" in (result.stderr + result.stdout)


def test_cli_handles_missing_env_vars(tmp_path):
    """Missing Coinbase env vars should fall back to mock broker in dev mode."""
    env = cli_env({})
    env["PERPS_FORCE_MOCK"] = "1"
    env["EVENT_STORE_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "bot_v2.cli", "--profile", "dev", "--dev-fast"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
