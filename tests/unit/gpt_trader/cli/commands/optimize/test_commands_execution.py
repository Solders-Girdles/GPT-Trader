"""Unit tests for optimize CLI command execution paths."""

from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gpt_trader.cli.commands import strategy_profile as strategy_cmd
from gpt_trader.cli.commands.optimize import apply, compare, export, run, view
from gpt_trader.cli.commands.optimize import list as list_cmd
from gpt_trader.cli.response import CliErrorCode, CliResponse


class TestRunCommand:
    """Test run command functionality."""

    def test_dry_run_returns_success(self):
        """Test dry run mode returns success."""
        args = Namespace(
            config=None,
            objective="sharpe",
            strategy="perps_baseline",
            symbols=["BTC-USD"],
            name=None,
            trials=10,
            sampler="tpe",
            seed=None,
            timeout=None,
            start_date=None,
            end_date=None,
            granularity="FIVE_MINUTE",
            format="text",
            quiet=False,
            dry_run=True,
        )

        result = run.execute(args)
        assert result == 0


class TestListCommand:
    """Test list command functionality."""

    def test_execute_with_no_runs(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """Test list command with no runs."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        monkeypatch.setattr(list_cmd, "OptimizationStorage", MagicMock(return_value=mock_storage))
        args = Namespace(format="text", limit=20, study=None)
        result = list_cmd.execute(args)

        assert result == 0

    def test_execute_with_runs(self, monkeypatch: pytest.MonkeyPatch):
        """Test list command with runs."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = [
            {
                "run_id": "opt_123",
                "study_name": "test",
                "started_at": "2024-01-01T00:00:00",
                "best_value": 1.5,
            }
        ]

        monkeypatch.setattr(list_cmd, "OptimizationStorage", MagicMock(return_value=mock_storage))
        args = Namespace(format="text", limit=20, study=None)
        result = list_cmd.execute(args)

        assert result == 0


class TestViewCommand:
    """Test view command functionality."""

    def test_execute_with_missing_run(self, monkeypatch: pytest.MonkeyPatch):
        """Test view command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        monkeypatch.setattr(view, "OptimizationStorage", MagicMock(return_value=mock_storage))
        args = Namespace(
            run_id="latest", format="text", trials=10, show_params=False, summary_only=False
        )
        result = view.execute(args)

        assert result == 1  # Error: no runs found


class TestCompareCommand:
    """Test compare command functionality."""

    def test_execute_requires_two_runs(self, capsys):
        """Test compare command requires at least 2 runs."""
        args = Namespace(run_ids=["opt_123"], format="text")
        result = compare.execute(args)
        assert result == 1

    def _mock_run(self, data: dict[str, Any]) -> MagicMock:
        run = MagicMock()
        run.to_dict.return_value = data
        return run

    def test_execute_json_returns_baseline_payload(self, monkeypatch: pytest.MonkeyPatch):
        """Test JSON output includes baseline metadata and matrix deltas."""
        run_a = {
            "run_id": "opt_a",
            "study_name": "alpha",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T01:00:00",
            "total_trials": 100,
            "feasible_trials": 90,
            "best_objective_value": 1.8,
            "best_parameters": {"p": 1},
        }
        run_b = {
            "run_id": "opt_b",
            "study_name": "beta",
            "started_at": "2024-01-02T00:00:00",
            "completed_at": "2024-01-02T01:00:00",
            "total_trials": 95,
            "feasible_trials": 88,
            "best_objective_value": 2.0,
            "best_parameters": {"p": 2},
        }

        mock_storage = MagicMock()
        mock_storage.load_run.side_effect = [
            self._mock_run(run_a),
            self._mock_run(run_b),
        ]

        monkeypatch.setattr(compare, "OptimizationStorage", MagicMock(return_value=mock_storage))

        args = Namespace(run_ids=["opt_a", "opt_b"], output_format="json", baseline="opt_b")
        response = compare.execute(args)

        assert isinstance(response, CliResponse)
        assert response.success
        assert response.data["baseline_run"]["run_id"] == "opt_b"
        matrix = response.data["matrix"]
        assert matrix[0]["values"][0]["delta"] == pytest.approx(1.8 - 2.0)
        assert matrix[0]["values"][1]["delta"] == pytest.approx(0.0)

    def test_execute_json_invalid_baseline_returns_error(self, monkeypatch: pytest.MonkeyPatch):
        """Test baseline validation rejects run IDs outside the comparison set."""
        run_a = {
            "run_id": "opt_a",
            "study_name": "alpha",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T01:00:00",
            "total_trials": 100,
            "feasible_trials": 90,
            "best_objective_value": 1.8,
        }
        run_b = {
            "run_id": "opt_b",
            "study_name": "beta",
            "started_at": "2024-01-02T00:00:00",
            "completed_at": "2024-01-02T01:00:00",
            "total_trials": 95,
            "feasible_trials": 88,
            "best_objective_value": 2.0,
        }

        mock_storage = MagicMock()
        mock_storage.load_run.side_effect = [
            self._mock_run(run_a),
            self._mock_run(run_b),
        ]

        monkeypatch.setattr(compare, "OptimizationStorage", MagicMock(return_value=mock_storage))

        args = Namespace(run_ids=["opt_a", "opt_b"], output_format="json", baseline="missing")
        response = compare.execute(args)

        assert isinstance(response, CliResponse)
        assert not response.success
        assert response.errors
        assert response.errors[0].code == CliErrorCode.INVALID_ARGUMENT.value


class TestExportCommand:
    """Test export command functionality."""

    def test_execute_with_missing_run(self, monkeypatch: pytest.MonkeyPatch):
        """Test export command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        monkeypatch.setattr(export, "OptimizationStorage", MagicMock(return_value=mock_storage))
        args = Namespace(
            run_id="latest",
            format="json",
            output=None,
            best_only=False,
            include_trials=False,
        )
        result = export.execute(args)

        assert result == 1  # Error: no runs found


class TestApplyCommand:
    """Test apply command functionality."""

    def test_execute_with_missing_run(self, monkeypatch: pytest.MonkeyPatch):
        """Test apply command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        monkeypatch.setattr(apply, "OptimizationStorage", MagicMock(return_value=mock_storage))
        args = Namespace(
            run_id="latest",
            output=None,
            profile="optimized",
            base_config=None,
            strategy_only=False,
            dry_run=False,
        )
        result = apply.execute(args)

        assert result == 1  # Error: no runs found

    def test_build_output_config_strategy_only_filters_groups(self):
        run_data = {
            "run_id": "opt_123",
            "study_name": "study",
            "best_objective_value": 1.23,
            "best_parameters": {
                "short_ma_period": 10,
                "target_leverage": 4,
                "slippage_bps": 5,
            },
        }

        output = apply._build_output_config(
            run_data,
            base_config={},
            profile_name="test",
            strategy_only=True,
        )

        assert output["strategy"]["short_ma_period"] == 10
        assert "risk" not in output
        assert "simulation" not in output

    def test_build_output_config_includes_all_groups(self):
        run_data = {
            "run_id": "opt_321",
            "study_name": "study",
            "best_objective_value": 2.46,
            "best_parameters": {
                "short_ma_period": 8,
                "target_leverage": 5,
                "slippage_bps": 10,
            },
        }

        output = apply._build_output_config(
            run_data,
            base_config={},
            profile_name="test",
            strategy_only=False,
        )

        assert output["strategy"]["short_ma_period"] == 8
        assert output["risk"]["target_leverage"] == 5
        assert output["simulation"]["slippage_bps"] == 10

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
