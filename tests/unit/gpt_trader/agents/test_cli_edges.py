from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest

from gpt_trader.agents import cli


def test_get_scripts_dir_points_to_agents_scripts() -> None:
    scripts_dir = cli._get_scripts_dir()

    assert scripts_dir.name == "agents"
    assert scripts_dir.parent.name == "scripts"
    assert (scripts_dir / "quality_gate.py").exists()


def test_run_script_missing_returns_one_and_stderr(tmp_path, capsys) -> None:
    scripts_dir = tmp_path / "agents"
    scripts_dir.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "_get_scripts_dir", lambda: scripts_dir)

    try:
        result = cli._run_script("missing.py")
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert result == 1
    assert "Script not found" in captured.err


def test_run_script_forwards_args_and_sets_cwd(tmp_path, monkeypatch) -> None:
    scripts_dir = tmp_path / "scripts" / "agents"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "quality_gate.py"
    script_path.write_text("# stub")

    monkeypatch.setattr(cli, "_get_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(sys, "argv", ["agent-check", "--format", "text"])

    run_mock = Mock()
    run_mock.return_value = Mock(returncode=7)
    monkeypatch.setattr(cli.subprocess, "run", run_mock)

    result = cli._run_script("quality_gate.py")

    run_mock.assert_called_once_with(
        [sys.executable, str(script_path), "--format", "text"],
        cwd=scripts_dir.parent.parent,
    )
    assert result == 7


@pytest.mark.parametrize(
    ("func", "script_name"),
    [
        (cli.check, "quality_gate.py"),
        (cli.impact, "change_impact.py"),
        (cli.map_deps, "dependency_graph.py"),
        (cli.tests, "generate_test_inventory.py"),
        (cli.risk, "query_risk_config.py"),
        (cli.naming, "naming_inventory.py"),
        (cli.regenerate, "regenerate_all.py"),
    ],
)
def test_entrypoints_call_run_script(func, script_name, monkeypatch) -> None:
    run_mock = Mock(return_value=0)
    monkeypatch.setattr(cli, "_run_script", run_mock)

    assert func() == 0
    run_mock.assert_called_once_with(script_name)
