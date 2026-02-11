"""Unit tests for optimize CLI argument parsing."""

from __future__ import annotations

import argparse

import pytest

from gpt_trader.cli.commands.optimize import apply, compare, export, resume, run, view
from gpt_trader.cli.commands.optimize import list as list_cmd


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

    def test_compare_parses_baseline(self, parser):
        """Test compare command parses baseline argument."""
        args = parser.parse_args(
            ["optimize", "compare", "opt_123", "opt_456", "--baseline", "opt_456"]
        )
        assert args.baseline == "opt_456"

    def test_export_parses_format(self, parser):
        """Test export command parses export format."""
        args = parser.parse_args(["optimize", "export", "--export-format", "csv"])
        assert args.export_format == "csv"

    def test_apply_parses_profile(self, parser):
        """Test apply command parses profile."""
        args = parser.parse_args(["optimize", "apply", "--profile", "prod"])
        assert args.profile == "prod"
