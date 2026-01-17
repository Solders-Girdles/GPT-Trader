from __future__ import annotations

from pathlib import Path

import pytest
from scripts.ops import canary_process


def test_parse_ps_output_filters_canary_profile() -> None:
    output = """
    100 1 100 uv run gpt-trader run --profile canary --dry-run
    101 100 100 /usr/bin/python gpt-trader run --profile canary --dry-run
    200 1 200 uv run gpt-trader run --profile canary_open --dry-run
    300 1 300 uv run gpt-trader run --profile canary --dry-run --tui
    400 1 400 uv run other
    """
    processes = canary_process.parse_ps_output(output)
    canary = canary_process.find_canary_processes(processes, "canary")
    assert [proc.pid for proc in canary] == [100, 101]


def test_stop_removes_stale_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_path = tmp_path / "var" / "run" / "canary.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("99999\n")

    monkeypatch.setattr(canary_process, "list_processes", lambda: [])

    exit_code = canary_process.stop_canary(
        runtime_root=tmp_path,
        profile="canary",
        timeout_seconds=1,
    )

    assert exit_code == 0
    assert pid_path.exists() is False


def test_start_refuses_when_running_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    running = [
        canary_process.ProcessInfo(
            pid=101,
            ppid=1,
            pgid=101,
            command="uv run gpt-trader run --profile canary --dry-run",
        )
    ]
    monkeypatch.setattr(canary_process, "list_processes", lambda: running)

    exit_code = canary_process.start_canary(
        runtime_root=Path("."),
        profile="canary",
        force=False,
        stop_timeout=1,
    )

    assert exit_code == 1


def test_is_canary_command_excludes_canary_open() -> None:
    assert (
        canary_process.is_canary_command(
            "uv run gpt-trader run --profile canary_open --dry-run",
            "canary",
        )
        is False
    )
