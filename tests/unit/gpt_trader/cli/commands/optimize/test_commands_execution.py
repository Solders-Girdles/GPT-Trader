"""Unit tests for optimize CLI command execution paths."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.cli.commands.optimize import apply, compare, export, run, view
from gpt_trader.cli.commands.optimize import list as list_cmd


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
