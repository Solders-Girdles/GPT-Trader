"""Tests for CLI option helpers."""

from __future__ import annotations

from argparse import ArgumentParser

from gpt_trader.app.config.profile_loader import DEFAULT_RUNTIME_PROFILE_NAME
from gpt_trader.cli.options import add_profile_option


def test_direct_profile_option_has_default() -> None:
    parser = ArgumentParser()
    add_profile_option(parser)
    args = parser.parse_args([])

    assert args.profile == DEFAULT_RUNTIME_PROFILE_NAME


def test_inherited_profile_option_suppresses_default() -> None:
    parser = ArgumentParser()
    add_profile_option(parser, inherit_from_parent=True)
    args = parser.parse_args([])

    assert not hasattr(args, "profile")
