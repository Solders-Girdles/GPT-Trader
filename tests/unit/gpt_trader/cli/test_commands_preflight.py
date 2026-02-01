"""Unit tests for the preflight CLI command."""

from __future__ import annotations

import argparse
from argparse import Namespace

from gpt_trader.cli.commands import preflight as preflight_cmd


def test_preflight_registers_parser() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    preflight_cmd.register(subparsers)

    assert "preflight" in subparsers.choices
    args = parser.parse_args(["preflight"])
    assert args.profile == "canary"
    assert args.verbose is False
    assert args.warn_only is False


def test_preflight_execute_delegates(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(preflight_cmd, "run_preflight_cli", fake_run)

    args = Namespace(profile="prod", verbose=True, warn_only=True)
    exit_code = preflight_cmd.execute(args)

    assert exit_code == 0
    assert captured["argv"] == ["--profile", "prod", "--verbose", "--warn-only"]
