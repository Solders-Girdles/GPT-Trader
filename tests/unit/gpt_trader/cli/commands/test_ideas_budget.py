from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.cli.response import CliErrorCode
from gpt_trader.features.trade_ideas import (
    ACTOR_ENV_VAR,
    DEFAULT_IDEAS_ROOT,
    IDEAS_ROOT_ENV_VAR,
    resolve_ideas_root,
    resolve_trade_idea_actor_id,
)


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, json.loads(output)


def _root_args(root: Path) -> list[str]:
    return ["--ideas-root", str(root), "--format", "json"]


def test_budget_show_seeds_defaults_and_budget_set_bumps_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"

    exit_code, response = _run_json(capsys, ["ideas", "budget", "show", *_root_args(root)])
    assert exit_code == 0
    assert response["data"]["version"] == 1
    assert response["data"]["max_loss_per_idea_pct"] == "5"

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "budget",
            "set",
            *_root_args(root),
            "--actor",
            "rj",
            "--max-loss-per-idea-pct",
            "2",
            "--reason",
            "Tighten canary budget",
        ],
    )
    assert exit_code == 0
    assert response["data"]["version"] == 2
    assert response["data"]["max_loss_per_idea_pct"] == "2"
    assert response["data"]["reason"] == "Tighten canary budget"


def test_budget_set_requires_at_least_one_field_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "budget",
            "set",
            *_root_args(root),
            "--actor",
            "rj",
            "--reason",
            "No changes",
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.MISSING_ARGUMENT.value


def test_shared_root_and_actor_resolution_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_root = tmp_path / "configured"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv(IDEAS_ROOT_ENV_VAR, str(configured_root))
    monkeypatch.setenv(ACTOR_ENV_VAR, "env-actor")

    assert resolve_ideas_root(explicit_root) == explicit_root
    assert resolve_ideas_root() == configured_root
    assert resolve_trade_idea_actor_id("flag-actor") == "flag-actor"
    assert resolve_trade_idea_actor_id(None) == "env-actor"

    monkeypatch.delenv(IDEAS_ROOT_ENV_VAR)
    monkeypatch.delenv(ACTOR_ENV_VAR)
    assert resolve_ideas_root() == DEFAULT_IDEAS_ROOT
