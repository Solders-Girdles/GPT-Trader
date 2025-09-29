"""CLI execution scenarios covering environment handling and quick runs."""

from __future__ import annotations

import subprocess
import sys
import tempfile

import pytest

from tests.unit.bot_v2.cli_test_utils import cli_env


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
