from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.ci.check_tui_css_up_to_date as check_tui_css_up_to_date


def test_project_root_includes_script() -> None:
    project_root = check_tui_css_up_to_date._project_root()
    script_path = project_root / "scripts" / "ci" / "check_tui_css_up_to_date.py"

    assert script_path.exists()


def test_run_build_invokes_script(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "scripts" / "build_tui_css.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# stub\n", encoding="utf-8")

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(args: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd, check))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(check_tui_css_up_to_date.subprocess, "run", fake_run)

    check_tui_css_up_to_date._run_build(tmp_path)

    assert calls == [([sys.executable, str(script_path)], tmp_path, True)]


def test_diff_names_filters_blank_lines(monkeypatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def fake_run(
        args: list[str],
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        recorded.update(
            {
                "args": args,
                "cwd": cwd,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(args, 0, stdout="src/a\n\nsrc/b\n")

    monkeypatch.setattr(check_tui_css_up_to_date.subprocess, "run", fake_run)

    names = check_tui_css_up_to_date._diff_names(tmp_path)

    assert names == ["src/a", "src/b"]
    assert recorded["args"][:3] == ["git", "diff", "--name-only"]
    assert recorded["cwd"] == tmp_path
    assert recorded["check"] is True
    assert recorded["capture_output"] is True
    assert recorded["text"] is True
    for path in check_tui_css_up_to_date.GENERATED_FILES:
        assert str(path) in recorded["args"]


def test_print_diff_calls_git(monkeypatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def fake_run(args: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        recorded.update({"args": args, "cwd": cwd, "check": check})
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(check_tui_css_up_to_date.subprocess, "run", fake_run)

    check_tui_css_up_to_date._print_diff(tmp_path)

    assert recorded["args"][:2] == ["git", "diff"]
    assert recorded["cwd"] == tmp_path
    assert recorded["check"] is False
    for path in check_tui_css_up_to_date.GENERATED_FILES:
        assert str(path) in recorded["args"]


def test_main_returns_zero_when_no_changes(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, Path] = {}

    def fake_project_root() -> Path:
        return tmp_path

    def fake_run_build(project_root: Path) -> None:
        called["run_build"] = project_root

    def fake_diff_names(project_root: Path) -> list[str]:
        called["diff_names"] = project_root
        return []

    monkeypatch.setattr(check_tui_css_up_to_date, "_project_root", fake_project_root)
    monkeypatch.setattr(check_tui_css_up_to_date, "_run_build", fake_run_build)
    monkeypatch.setattr(check_tui_css_up_to_date, "_diff_names", fake_diff_names)

    result = check_tui_css_up_to_date.main()

    assert result == 0
    assert called["run_build"] == tmp_path
    assert called["diff_names"] == tmp_path


def test_main_reports_outdated_css(monkeypatch, capsys, tmp_path: Path) -> None:
    changed = [
        "src/gpt_trader/tui/styles/main.tcss",
        "src/gpt_trader/tui/styles/main_light.tcss",
    ]

    def fake_project_root() -> Path:
        return tmp_path

    def fake_run_build(project_root: Path) -> None:
        return None

    def fake_diff_names(project_root: Path) -> list[str]:
        return changed

    recorded: dict[str, Path] = {}

    def fake_print_diff(project_root: Path) -> None:
        recorded["print_diff"] = project_root

    monkeypatch.setattr(check_tui_css_up_to_date, "_project_root", fake_project_root)
    monkeypatch.setattr(check_tui_css_up_to_date, "_run_build", fake_run_build)
    monkeypatch.setattr(check_tui_css_up_to_date, "_diff_names", fake_diff_names)
    monkeypatch.setattr(check_tui_css_up_to_date, "_print_diff", fake_print_diff)

    result = check_tui_css_up_to_date.main()
    output = capsys.readouterr().out

    assert result == 1
    assert "Error: TUI CSS is out of date." in output
    assert "Run `python scripts/build_tui_css.py` and commit the updated files:" in output
    for name in changed:
        assert f"  - {name}" in output
    assert recorded["print_diff"] == tmp_path
