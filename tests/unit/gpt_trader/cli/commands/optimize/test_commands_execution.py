"""Unit tests for optimize CLI command execution paths."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.cli.commands.optimize import (
    apply,
    artifact_activate,
    artifact_publish,
    compare,
    export,
    run,
    view,
)
from gpt_trader.cli.commands.optimize import list as list_cmd
from gpt_trader.cli.response import CliResponse
from gpt_trader.features.research.artifacts import StrategyArtifact, StrategyArtifactStore


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

    def test_config_file_is_not_overridden_by_defaults(self, tmp_path):
        """Config file values should remain unless CLI overrides are explicit."""
        config_path = tmp_path / "optimize.yml"
        config_path.write_text(
            "\n".join(
                [
                    "study:",
                    "  name: cfg_run",
                    "  trials: 3",
                    "  sampler: random",
                    "objective:",
                    "  preset: total_return",
                    "strategy:",
                    "  type: spot",
                    "  symbols:",
                    "    - ETH-USD",
                    "backtest:",
                    "  start_date: \"2026-01-01\"",
                    "  end_date: \"2026-01-02\"",
                    "  granularity: \"ONE_HOUR\"",
                ]
            )
            + "\n"
        )

        args = Namespace(
            config=config_path,
            objective=None,
            strategy=None,
            symbols=None,
            name=None,
            trials=None,
            sampler=None,
            seed=None,
            timeout=None,
            start_date=None,
            end_date=None,
            granularity=None,
            format="text",
            quiet=True,
            dry_run=True,
        )

        config = run._build_config_from_args(args)

        assert config.study.name == "cfg_run"
        assert config.study.trials == 3
        assert config.study.sampler == "random"
        assert config.objective_name == "total_return"
        assert config.strategy_type == "baseline"
        assert config.strategy_variant == "spot"
        assert config.symbols == ["ETH-USD"]
        assert config.backtest is not None
        assert config.backtest.granularity == "ONE_HOUR"

    def test_cli_overrides_apply_over_config(self, tmp_path):
        """Explicit CLI flags should override config file values."""
        config_path = tmp_path / "optimize.yml"
        config_path.write_text(
            "\n".join(
                [
                    "study:",
                    "  name: cfg_run",
                    "  trials: 3",
                    "  sampler: random",
                    "objective:",
                    "  preset: total_return",
                    "strategy:",
                    "  type: spot",
                    "  symbols:",
                    "    - ETH-USD",
                    "backtest:",
                    "  start_date: \"2026-01-01\"",
                    "  end_date: \"2026-01-02\"",
                    "  granularity: \"ONE_HOUR\"",
                ]
            )
            + "\n"
        )

        args = Namespace(
            config=config_path,
            objective="sharpe",
            strategy="perps_baseline",
            symbols=["BTC-USD", "SOL-USD"],
            name=None,
            trials=12,
            sampler="tpe",
            seed=42,
            timeout=3600,
            start_date="2026-01-05",
            end_date="2026-01-06",
            granularity="FIVE_MINUTE",
            format="text",
            quiet=True,
            dry_run=True,
        )

        config = run._build_config_from_args(args)

        assert config.study.name == "cfg_run"
        assert config.study.trials == 12
        assert config.study.sampler == "tpe"
        assert config.study.seed == 42
        assert config.study.timeout_seconds == 3600
        assert config.objective_name == "sharpe"
        assert config.strategy_type == "baseline"
        assert config.strategy_variant == "perps"
        assert config.symbols == ["BTC-USD", "SOL-USD"]
        assert config.backtest is not None
        assert config.backtest.start_date.isoformat().startswith("2026-01-05")
        assert config.backtest.end_date.isoformat().startswith("2026-01-06")
        assert config.backtest.granularity == "FIVE_MINUTE"


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


class TestArtifactCommands:
    """Test artifact publish/activate commands."""

    def test_publish_artifact_marks_approved(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
        store = StrategyArtifactStore()
        artifact = StrategyArtifact.create(
            strategy_type="baseline",
            symbols=["BTC-USD"],
            interval=60,
        )
        store.save(artifact)

        args = Namespace(
            artifact=artifact.artifact_id,
            approved_by="tester",
            notes="ok",
            format="json",
            output_format="json",
        )

        result = artifact_publish.execute(args)
        assert isinstance(result, CliResponse)
        assert result.success is True

        reloaded = store.load(artifact.artifact_id)
        assert reloaded.approved is True
        assert reloaded.approved_by == "tester"

    def test_activate_artifact_sets_active(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
        store = StrategyArtifactStore()
        artifact = StrategyArtifact.create(
            strategy_type="baseline",
            symbols=["BTC-USD"],
            interval=60,
        )
        store.save(artifact)
        store.publish(artifact.artifact_id, approved_by="tester")

        args = Namespace(
            artifact=artifact.artifact_id,
            profile="prod",
            allow_unapproved=False,
            format="json",
            output_format="json",
        )

        result = artifact_activate.execute(args)
        assert isinstance(result, CliResponse)
        assert result.success is True
        assert store.resolve_active("prod") == artifact.artifact_id

    def test_activate_artifact_rejects_unapproved(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
        store = StrategyArtifactStore()
        artifact = StrategyArtifact.create(
            strategy_type="baseline",
            symbols=["BTC-USD"],
            interval=60,
        )
        store.save(artifact)

        args = Namespace(
            artifact=artifact.artifact_id,
            profile="prod",
            allow_unapproved=False,
            format="json",
            output_format="json",
        )

        result = artifact_activate.execute(args)
        assert isinstance(result, CliResponse)
        assert result.success is False
        assert result.errors[0].code == "VALIDATION_ERROR"

    def test_activate_artifact_allows_unapproved_with_warning(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
        store = StrategyArtifactStore()
        artifact = StrategyArtifact.create(
            strategy_type="baseline",
            symbols=["BTC-USD"],
            interval=60,
        )
        store.save(artifact)

        args = Namespace(
            artifact=artifact.artifact_id,
            profile="prod",
            allow_unapproved=True,
            format="json",
            output_format="json",
        )

        result = artifact_activate.execute(args)
        assert isinstance(result, CliResponse)
        assert result.success is True
        assert result.warnings == ["Activating unapproved artifact"]
