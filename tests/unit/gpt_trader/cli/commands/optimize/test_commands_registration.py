"""Unit tests for optimize CLI command registration."""

from __future__ import annotations

import argparse

import pytest

from gpt_trader.cli.commands.optimize import (
    apply,
    artifact_activate,
    artifact_publish,
    compare,
    export,
    resume,
    run,
    view,
)
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

    def test_artifact_publish_register(self, subparsers):
        """Test artifact-publish subcommand registers."""
        artifact_publish.register(subparsers)
        parser = subparsers.choices.get("artifact-publish")
        assert parser is not None

    def test_artifact_activate_register(self, subparsers):
        """Test artifact-activate subcommand registers."""
        artifact_activate.register(subparsers)
        parser = subparsers.choices.get("artifact-activate")
        assert parser is not None
