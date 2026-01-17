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


def test_start_force_aborts_when_stop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    running = [
        canary_process.ProcessInfo(
            pid=101,
            ppid=1,
            pgid=101,
            command="uv run gpt-trader run --profile canary --dry-run",
        )
    ]
    monkeypatch.setattr(canary_process, "list_processes", lambda: running)
    monkeypatch.setattr(canary_process, "stop_canary", lambda **_: 1)

    def _raise_spawn(**_: object) -> int:
        raise AssertionError("spawn should not be called")

    monkeypatch.setattr(canary_process, "_spawn_canary", _raise_spawn)

    exit_code = canary_process.start_canary(
        runtime_root=Path("."),
        profile="canary",
        force=True,
        stop_timeout=1,
    )

    assert exit_code == 1


def test_restart_aborts_when_stop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canary_process, "stop_canary", lambda **_: 1)

    exit_code = canary_process.restart_canary(
        runtime_root=Path("."),
        profile="canary",
        force=False,
        stop_timeout=1,
        wait_seconds=1,
        wait_interval=1,
    )

    assert exit_code == 1


def test_restart_passes_baselines_to_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canary_process, "_fetch_latest_event_id", lambda *_: 10)
    monkeypatch.setattr(canary_process, "stop_canary", lambda **_: 0)
    monkeypatch.setattr(canary_process, "start_canary", lambda **_: 0)
    monkeypatch.setattr(canary_process, "_resolve_build_sha", lambda *_: "sha")

    commands: list[list[str]] = []

    def _capture_wait(*, command: list[str], **_: object) -> bool:
        commands.append(command)
        return True

    monkeypatch.setattr(canary_process, "_wait_for_success", _capture_wait)

    exit_code = canary_process.restart_canary(
        runtime_root=Path("."),
        profile="canary",
        force=False,
        stop_timeout=1,
        wait_seconds=1,
        wait_interval=1,
    )

    assert exit_code == 0
    assert any("--min-event-id" in cmd for cmd in commands)
    assert any("--expected-build-sha" in cmd for cmd in commands)
    assert any("runtime_fingerprint.py" in " ".join(cmd) for cmd in commands)


def test_is_canary_command_excludes_canary_open() -> None:
    assert (
        canary_process.is_canary_command(
            "uv run gpt-trader run --profile canary_open --dry-run",
            "canary",
        )
        is False
    )
