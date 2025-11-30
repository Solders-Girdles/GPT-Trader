"""Unit tests for optimize CLI commands."""

from __future__ import annotations

import argparse
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.cli.commands.optimize import apply, compare, export, resume, run, view
from gpt_trader.cli.commands.optimize import list as list_cmd


class TestOptimizeSubcommandRegistration:
    """Test that all subcommands register correctly."""

    @pytest.fixture
    def subparsers(self):
        """Create subparsers for testing."""
        parser = argparse.ArgumentParser()
        return parser.add_subparsers(dest="command")

    def test_run_register(self, subparsers):
        """Test run subcommand registers."""
        run.register(subparsers)
        parser = subparsers.choices.get("run")
        assert parser is not None

    def test_view_register(self, subparsers):
        """Test view subcommand registers."""
        view.register(subparsers)
        parser = subparsers.choices.get("view")
        assert parser is not None

    def test_list_register(self, subparsers):
        """Test list subcommand registers."""
        list_cmd.register(subparsers)
        parser = subparsers.choices.get("list")
        assert parser is not None

    def test_compare_register(self, subparsers):
        """Test compare subcommand registers."""
        compare.register(subparsers)
        parser = subparsers.choices.get("compare")
        assert parser is not None

    def test_export_register(self, subparsers):
        """Test export subcommand registers."""
        export.register(subparsers)
        parser = subparsers.choices.get("export")
        assert parser is not None

    def test_resume_register(self, subparsers):
        """Test resume subcommand registers."""
        resume.register(subparsers)
        parser = subparsers.choices.get("resume")
        assert parser is not None

    def test_apply_register(self, subparsers):
        """Test apply subcommand registers."""
        apply.register(subparsers)
        parser = subparsers.choices.get("apply")
        assert parser is not None


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

    def test_execute_with_no_runs(self, tmp_path, monkeypatch):
        """Test list command with no runs."""
        # Patch storage to use temp directory
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        with patch.object(list_cmd, "OptimizationStorage", return_value=mock_storage):
            args = Namespace(format="text", limit=20, study=None)
            result = list_cmd.execute(args)

        assert result == 0

    def test_execute_with_runs(self, monkeypatch):
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

        with patch.object(list_cmd, "OptimizationStorage", return_value=mock_storage):
            args = Namespace(format="text", limit=20, study=None)
            result = list_cmd.execute(args)

        assert result == 0


class TestViewCommand:
    """Test view command functionality."""

    def test_execute_with_missing_run(self, monkeypatch):
        """Test view command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        with patch.object(view, "OptimizationStorage", return_value=mock_storage):
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

    def test_execute_with_missing_run(self, monkeypatch):
        """Test export command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        with patch.object(export, "OptimizationStorage", return_value=mock_storage):
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

    def test_execute_with_missing_run(self, monkeypatch):
        """Test apply command with missing run."""
        mock_storage = MagicMock()
        mock_storage.list_runs.return_value = []

        with patch.object(apply, "OptimizationStorage", return_value=mock_storage):
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


class TestArgumentParsing:
    """Test argument parsing for all commands."""

    @pytest.fixture
    def parser(self):
        """Create a complete parser with all subcommands."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(dest="command")

        opt_parser = subparsers.add_parser("optimize")
        opt_subparsers = opt_parser.add_subparsers(dest="subcommand")

        run.register(opt_subparsers)
        view.register(opt_subparsers)
        list_cmd.register(opt_subparsers)
        compare.register(opt_subparsers)
        export.register(opt_subparsers)
        resume.register(opt_subparsers)
        apply.register(opt_subparsers)

        return main_parser

    def test_run_parses_objective(self, parser):
        """Test run command parses objective."""
        args = parser.parse_args(["optimize", "run", "--objective", "sortino"])
        assert args.objective == "sortino"

    def test_run_parses_trials(self, parser):
        """Test run command parses trials."""
        args = parser.parse_args(["optimize", "run", "--trials", "50"])
        assert args.trials == 50

    def test_run_parses_symbols(self, parser):
        """Test run command parses symbols."""
        args = parser.parse_args(["optimize", "run", "--symbols", "BTC-USD", "ETH-USD"])
        assert args.symbols == ["BTC-USD", "ETH-USD"]

    def test_view_parses_run_id(self, parser):
        """Test view command parses run ID."""
        args = parser.parse_args(["optimize", "view", "opt_123"])
        assert args.run_id == "opt_123"

    def test_list_parses_limit(self, parser):
        """Test list command parses limit."""
        args = parser.parse_args(["optimize", "list", "--limit", "5"])
        assert args.limit == 5

    def test_compare_parses_run_ids(self, parser):
        """Test compare command parses run IDs."""
        args = parser.parse_args(["optimize", "compare", "opt_123", "opt_456"])
        assert args.run_ids == ["opt_123", "opt_456"]

    def test_export_parses_format(self, parser):
        """Test export command parses export format."""
        args = parser.parse_args(["optimize", "export", "--export-format", "csv"])
        assert args.export_format == "csv"

    def test_apply_parses_profile(self, parser):
        """Test apply command parses profile."""
        args = parser.parse_args(["optimize", "apply", "--profile", "prod"])
        assert args.profile == "prod"
