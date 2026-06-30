"""Unit tests for the `strategy profile-diff` CLI command.

Extracted from the optimize command-execution suite: this exercises
``gpt_trader.cli.commands.strategy_profile``, not the optimize subcommands.
"""

from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from gpt_trader.cli.commands import strategy_profile as strategy_cmd
from gpt_trader.cli.response import CliErrorCode, CliResponse


class TestStrategyProfileDiffCommand:
    """Tests for the strategy profile diff CLI."""

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w") as f:
            json.dump(data, f)

    def test_execute_returns_diff_entries(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = tmp_path / "baseline.json"
        runtime = tmp_path / "runtime.json"
        self._write_json(baseline, {"name": "alpha", "risk": {"max": 0.1}})
        self._write_json(runtime, {"name": "alpha", "risk": {"max": 0.2}})

        args = Namespace(
            baseline=str(baseline),
            runtime_profile=str(runtime),
            runtime_root=str(tmp_path),
            profile="dev",
            ignore_fields=[],
            output_format="json",
            output=None,
            quiet=False,
        )

        response = strategy_cmd.execute_profile_diff(args)
        assert isinstance(response, CliResponse)
        assert response.success
        diff = response.data["diff"]
        assert any(entry["path"] == "risk.max" and entry["status"] == "changed" for entry in diff)

    def test_execute_missing_runtime_profile_returns_error(self, tmp_path) -> None:
        baseline = tmp_path / "baseline.json"
        self._write_json(baseline, {"name": "alpha"})
        runtime = tmp_path / "missing.json"

        args = Namespace(
            baseline=str(baseline),
            runtime_profile=str(runtime),
            runtime_root=str(tmp_path),
            profile="dev",
            ignore_fields=[],
            output_format="json",
            output=None,
            quiet=False,
        )

        response = strategy_cmd.execute_profile_diff(args)
        assert isinstance(response, CliResponse)
        assert not response.success
        assert response.errors
        assert response.errors[0].code == CliErrorCode.FILE_NOT_FOUND.value

    def test_cli_profile_diff_smoke_json_output(self, tmp_path, capsys) -> None:
        baseline = tmp_path / "baseline.json"
        runtime = tmp_path / "runtime.json"
        self._write_json(baseline, {"name": "alpha", "risk": {"max": 0.1}})
        self._write_json(runtime, {"name": "alpha", "risk": {"max": 0.2}})

        cli = importlib.import_module("gpt_trader.cli")
        argv = [
            "strategy",
            "profile-diff",
            "--baseline",
            str(baseline),
            "--runtime-profile",
            str(runtime),
            "--format",
            "json",
        ]

        exit_code = cli.main(argv)
        assert exit_code == 0

        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True
        assert output["exit_code"] == 0
        assert output["command"] == strategy_cmd.COMMAND_NAME
        data = output["data"]
        assert data["baseline_path"] == str(baseline)
        assert data["runtime_profile_path"] == str(runtime)
        assert isinstance(data["diff"], list)
        assert any(
            entry["path"] == "risk.max" and entry["status"] == "changed" for entry in data["diff"]
        )

    def test_cli_profile_diff_missing_runtime_profile_returns_error(self, tmp_path, capsys) -> None:
        baseline = tmp_path / "baseline.json"
        self._write_json(baseline, {"name": "alpha"})
        missing_runtime = tmp_path / "missing.json"

        cli = importlib.import_module("gpt_trader.cli")
        argv = [
            "strategy",
            "profile-diff",
            "--baseline",
            str(baseline),
            "--runtime-profile",
            str(missing_runtime),
            "--format",
            "json",
        ]

        exit_code = cli.main(argv)
        assert exit_code == 1

        output = json.loads(capsys.readouterr().out)
        assert output["success"] is False
        assert output["exit_code"] == 1
        assert output["command"] == strategy_cmd.COMMAND_NAME
        assert output["data"] is None
        errors = output["errors"]
        assert errors
        assert errors[0]["code"] == CliErrorCode.FILE_NOT_FOUND.value
        assert errors[0]["details"]["path"] == str(missing_runtime)

    def test_execute_uses_first_existing_default_runtime_profile_path(self, tmp_path) -> None:
        baseline = tmp_path / "baseline.json"
        self._write_json(baseline, {"name": "alpha", "risk": {"max": 0.1}})

        runtime_profile = tmp_path / "config" / "profiles" / "dev.yaml"
        runtime_profile.parent.mkdir(parents=True, exist_ok=True)
        runtime_profile.write_text("name: alpha\nrisk:\n  max: 0.2\n")

        args = Namespace(
            baseline=str(baseline),
            runtime_profile=None,
            runtime_root=str(tmp_path),
            profile="dev",
            ignore_fields=[],
            output_format="json",
            output=None,
            quiet=False,
        )

        response = strategy_cmd.execute_profile_diff(args)
        assert isinstance(response, CliResponse)
        assert response.success
        assert response.data["runtime_profile_path"] == str(runtime_profile)
