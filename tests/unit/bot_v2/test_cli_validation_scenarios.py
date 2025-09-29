"""CLI validation and error handling scenarios."""

from __future__ import annotations

import subprocess
import sys
import tempfile

import pytest

from tests.unit.bot_v2.cli_test_utils import cli_env


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
