#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

DEFAULT_PROFILE = "canary"
DEFAULT_STOP_TIMEOUT = 10
DEFAULT_WAIT_SECONDS = 90
DEFAULT_WAIT_INTERVAL = 5


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    pgid: int
    command: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage canary process lifecycle.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name (default: canary)",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force start even if canary is already running",
    )
    parser.add_argument(
        "--stop-timeout",
        type=int,
        default=DEFAULT_STOP_TIMEOUT,
        help="Seconds to wait before SIGKILL (default: 10)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait for post-start checks (default: 90)",
    )
    parser.add_argument(
        "--wait-interval",
        type=int,
        default=DEFAULT_WAIT_INTERVAL,
        help="Seconds between post-start checks (default: 5)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "stop", "start", "restart"):
        subparsers.add_parser(name)

    args = parser.parse_args()
    return args


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _extract_profile(tokens: list[str]) -> str | None:
    for idx, token in enumerate(tokens):
        if token == "--profile" and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith("--profile="):
            return token.split("=", 1)[1]
    return None


def _is_gpt_trader_run(tokens: list[str]) -> bool:
    for idx, token in enumerate(tokens):
        if "gpt-trader" in token and idx + 1 < len(tokens):
            return tokens[idx + 1] == "run"
    return False


def _is_tui_or_demo(tokens: list[str]) -> bool:
    return "--tui" in tokens or "--demo" in tokens


def is_canary_command(command: str, profile: str) -> bool:
    tokens = _command_tokens(command)
    if _is_tui_or_demo(tokens):
        return False
    if not _is_gpt_trader_run(tokens):
        return False
    return _extract_profile(tokens) == profile


def parse_ps_output(output: str) -> list[ProcessInfo]:
    processes: list[ProcessInfo] = []
    for line in output.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=3)
        if len(parts) < 4:
            continue
        pid, ppid, pgid, command = parts
        try:
            processes.append(
                ProcessInfo(
                    pid=int(pid),
                    ppid=int(ppid),
                    pgid=int(pgid),
                    command=command,
                )
            )
        except ValueError:
            continue
    return processes


def list_processes() -> list[ProcessInfo]:
    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ps failed")
    return parse_ps_output(result.stdout)


def find_canary_processes(
    processes: Iterable[ProcessInfo],
    profile: str,
) -> list[ProcessInfo]:
    return [process for process in processes if is_canary_command(process.command, profile)]


def read_pid_file(pid_path: Path) -> int | None:
    try:
        data = pid_path.read_text().strip()
    except FileNotFoundError:
        return None
    if not data:
        return None
    try:
        return int(data)
    except ValueError:
        return None


def write_pid_file(pid_path: Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n")


def _remove_pid_file(pid_path: Path) -> None:
    try:
        pid_path.unlink()
    except FileNotFoundError:
        return


def _format_pid_list(pids: Iterable[int]) -> str:
    return ",".join(str(pid) for pid in sorted(pids)) or "-"


def _resolve_build_sha(runtime_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runtime_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "git rev-parse failed"
        raise RuntimeError(message)
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse returned empty SHA")
    return sha


def _spawn_canary(
    *,
    runtime_root: Path,
    profile: str,
    build_sha: str,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root.resolve()
    env = os.environ.copy()
    env["GPT_TRADER_BUILD_SHA"] = build_sha

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [
                "uv",
                "run",
                "gpt-trader",
                "run",
                "--profile",
                profile,
                "--dry-run",
            ],
            cwd=runtime_root,
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    return process.pid


def _wait_for_exit(pgids: set[int], timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        running = [process for process in list_processes() if process.pgid in pgids]
        if not running:
            return True
        time.sleep(1)
    return False


def stop_canary(
    *,
    runtime_root: Path,
    profile: str,
    timeout_seconds: int,
) -> int:
    pid_path = runtime_root / "var" / "run" / "canary.pid"
    processes = list_processes()
    canary_processes = find_canary_processes(processes, profile)
    target_pgids = {proc.pgid for proc in canary_processes}

    pid_file_pid = read_pid_file(pid_path)
    pid_file_status = "missing"
    if pid_file_pid is not None:
        pid_file_status = "stale"
        pid_match = next(
            (proc for proc in processes if proc.pid == pid_file_pid),
            None,
        )
        if pid_match and is_canary_command(pid_match.command, profile):
            pid_file_status = "active"
            target_pgids.add(pid_match.pgid)

    if not target_pgids:
        _remove_pid_file(pid_path)
        print("status=stopped")
        print(f"pid_file_status={pid_file_status}")
        print("processes=0")
        return 0

    for pgid in sorted(target_pgids):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"error=failed to SIGTERM pgid {pgid}: {exc}")
            return 2

    if not _wait_for_exit(target_pgids, timeout_seconds):
        for pgid in sorted(target_pgids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                print(f"error=failed to SIGKILL pgid {pgid}: {exc}")
                return 2
        if not _wait_for_exit(target_pgids, timeout_seconds):
            print("error=canary processes still running")
            return 1

    _remove_pid_file(pid_path)
    print("status=stopped")
    print(f"pid_file_status={pid_file_status}")
    print(f"pgids={_format_pid_list(target_pgids)}")
    return 0


def start_canary(
    *,
    runtime_root: Path,
    profile: str,
    force: bool,
    stop_timeout: int,
) -> int:
    running = find_canary_processes(list_processes(), profile)
    if running and not force:
        print("status=running")
        print(f"pids={_format_pid_list(proc.pid for proc in running)}")
        return 1
    if running and force:
        stop_canary(runtime_root=runtime_root, profile=profile, timeout_seconds=stop_timeout)

    try:
        build_sha = _resolve_build_sha(runtime_root)
    except RuntimeError as exc:
        print(f"error={exc}")
        return 2

    log_path = runtime_root / "var" / "log" / "canary.out"
    pid_path = runtime_root / "var" / "run" / "canary.pid"
    pid = _spawn_canary(
        runtime_root=runtime_root,
        profile=profile,
        build_sha=build_sha,
        log_path=log_path,
    )
    write_pid_file(pid_path, pid)
    print("status=started")
    print(f"pid={pid}")
    print(f"build_sha={build_sha}")
    print(f"pid_file={pid_path}")
    print(f"log_file={log_path}")
    return 0


def _run_command(command: list[str], runtime_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=runtime_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _wait_for_success(
    *,
    command: list[str],
    runtime_root: Path,
    wait_seconds: int,
    wait_interval: int,
    label: str,
) -> bool:
    deadline = time.time() + wait_seconds
    last_result: subprocess.CompletedProcess[str] | None = None
    while time.time() <= deadline:
        last_result = _run_command(command, runtime_root)
        if last_result.returncode == 0:
            stdout = (last_result.stdout or "").strip()
            stderr = (last_result.stderr or "").strip()
            if stdout:
                print(stdout)
            if stderr:
                print(stderr)
            return True
        time.sleep(wait_interval)

    print(f"error=timeout waiting for {label}")
    if last_result is not None:
        stdout = (last_result.stdout or "").strip()
        stderr = (last_result.stderr or "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
    return False


def restart_canary(
    *,
    runtime_root: Path,
    profile: str,
    force: bool,
    stop_timeout: int,
    wait_seconds: int,
    wait_interval: int,
) -> int:
    stop_canary(runtime_root=runtime_root, profile=profile, timeout_seconds=stop_timeout)
    start_result = start_canary(
        runtime_root=runtime_root,
        profile=profile,
        force=force,
        stop_timeout=stop_timeout,
    )
    if start_result != 0:
        return start_result

    liveness_ok = _wait_for_success(
        command=[
            "uv",
            "run",
            "python",
            "scripts/ops/liveness_check.py",
            "--profile",
            profile,
            "--event-type",
            "heartbeat",
            "--event-type",
            "price_tick",
            "--max-age-seconds",
            "300",
        ],
        runtime_root=runtime_root,
        wait_seconds=wait_seconds,
        wait_interval=wait_interval,
        label="liveness_check",
    )

    runtime_ok = _wait_for_success(
        command=[
            "uv",
            "run",
            "python",
            "scripts/ops/runtime_fingerprint.py",
            "--profile",
            profile,
        ],
        runtime_root=runtime_root,
        wait_seconds=wait_seconds,
        wait_interval=wait_interval,
        label="runtime_start",
    )

    return 0 if liveness_ok and runtime_ok else 1


def status_canary(*, runtime_root: Path, profile: str) -> int:
    pid_path = runtime_root / "var" / "run" / "canary.pid"
    processes = list_processes()
    canary_processes = find_canary_processes(processes, profile)
    print("status=running" if canary_processes else "status=stopped")
    print(f"pids={_format_pid_list(proc.pid for proc in canary_processes)}")
    print(f"pgids={_format_pid_list(proc.pgid for proc in canary_processes)}")
    if pid_path.exists():
        pid_from_file = read_pid_file(pid_path)
        print(f"pid_file={pid_path}")
        print(f"pid_file_pid={pid_from_file or '-'}")
    return 0 if canary_processes else 1


def main() -> int:
    args = _parse_args()
    runtime_root = args.runtime_root.resolve()

    if args.command == "status":
        return status_canary(runtime_root=runtime_root, profile=args.profile)
    if args.command == "stop":
        return stop_canary(
            runtime_root=runtime_root,
            profile=args.profile,
            timeout_seconds=args.stop_timeout,
        )
    if args.command == "start":
        return start_canary(
            runtime_root=runtime_root,
            profile=args.profile,
            force=args.force,
            stop_timeout=args.stop_timeout,
        )
    if args.command == "restart":
        return restart_canary(
            runtime_root=runtime_root,
            profile=args.profile,
            force=args.force,
            stop_timeout=args.stop_timeout,
            wait_seconds=args.wait_seconds,
            wait_interval=args.wait_interval,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
