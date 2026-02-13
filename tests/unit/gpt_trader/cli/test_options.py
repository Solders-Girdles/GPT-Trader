"""Tests for CLI option helpers."""

from __future__ import annotations

from argparse import ArgumentParser

from gpt_trader.cli.options import add_profile_option


def test_direct_profile_option_has_default() -> None:
    parser = ArgumentParser()
    add_profile_option(parser)
    args = parser.parse_args([])

    assert args.profile is None


def test_inherited_profile_option_suppresses_default() -> None:
    parser = ArgumentParser()
    add_profile_option(parser, inherit_from_parent=True)
    args = parser.parse_args([])

    assert not hasattr(args, "profile")
