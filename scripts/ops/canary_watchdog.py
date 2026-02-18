#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import canary_process, liveness_check  # noqa: E402

STATE_FILE_NAME = "watchdog_state.json"


@dataclass
class WatchdogState:
    consecutive_reds: int = 0
    last_restart_ts: float | None = None


@dataclass
class PollOutcome:
    is_green: bool
    decision: str
    restart_failed: bool
    status_line: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch canary liveness and optionally restart on sustained RED."
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        default=None,
        help="Event type to consider (repeatable)",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=liveness_check.DEFAULT_MAX_AGE_SECONDS,
        help="Max age in seconds before marking RED (default: 300)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Seconds between polls (default: 60)",
    )
    parser.add_argument(
        "--restart-after-reds",
        type=int,
        default=2,
        help="Consecutive RED polls required before restart (default: 2)",
    )
    parser.add_argument(
        "--restart-cooldown-seconds",
        type=int,
        default=900,
        help="Minimum seconds between restarts (default: 900)",
    )
    parser.add_argument(
        "--auto-restart",
        action="store_true",
        help="Enable automatic canary restart on sustained RED",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and exit 0/1",
    )
    return parser.parse_args()


def _now() -> float:
    return time.time()


def _state_path(runtime_root: Path, profile: str) -> Path:
    return runtime_root / "runtime_data" / profile / STATE_FILE_NAME


def _load_state(state_path: Path) -> WatchdogState:
    try:
        payload = state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return WatchdogState()
    except (OSError, UnicodeDecodeError):
        return WatchdogState()

    try:
        raw_state = json.loads(payload)
    except json.JSONDecodeError:
        return WatchdogState()

    if not isinstance(raw_state, dict):
        return WatchdogState()

    consecutive_reds_raw = raw_state.get("consecutive_reds", 0)
    last_restart_ts_raw = raw_state.get("last_restart_ts")

    try:
        consecutive_reds = int(consecutive_reds_raw)
    except (TypeError, ValueError):
        consecutive_reds = 0
    if consecutive_reds < 0:
        consecutive_reds = 0

    last_restart_ts: float | None
    if last_restart_ts_raw is None:
        last_restart_ts = None
    else:
        try:
            last_restart_ts = float(last_restart_ts_raw)
        except (TypeError, ValueError):
            last_restart_ts = None

    return WatchdogState(
        consecutive_reds=consecutive_reds,
        last_restart_ts=last_restart_ts,
    )


def _save_state(state_path: Path, state: WatchdogState) -> None:
    payload = {
        "consecutive_reds": state.consecutive_reds,
        "last_restart_ts": state.last_restart_ts,
        "updated_at": _now(),
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _format_event_ages(rows: Iterable[liveness_check.EventAge]) -> str:
    parts = [f"{row.event_type}:{row.age_seconds}s" for row in rows]
    return ",".join(parts) if parts else "-"


def _poll_once(
    *,
    runtime_root: Path,
    profile: str,
    event_types: Iterable[str],
    max_age_seconds: int,
    auto_restart: bool,
    restart_after_reds: int,
    restart_cooldown_seconds: int,
    state: WatchdogState,
) -> PollOutcome:
    events_db = runtime_root / "runtime_data" / profile / "events.db"
    rows, is_green = liveness_check.check_liveness(
        events_db,
        event_types,
        max_age_seconds,
    )

    decision = "ok" if is_green else "red"
    restart_failed = False
    if is_green:
        state.consecutive_reds = 0
    else:
        state.consecutive_reds = min(state.consecutive_reds + 1, restart_after_reds)
        if auto_restart:
            if state.consecutive_reds >= restart_after_reds:
                now_ts = _now()
                if state.last_restart_ts is not None:
                    cooldown_remaining = restart_cooldown_seconds - (now_ts - state.last_restart_ts)
                else:
                    cooldown_remaining = 0
                if cooldown_remaining > 0:
                    decision = f"cooldown_remaining={int(cooldown_remaining)}s"
                else:
                    decision = "restart_triggered"
                    result = canary_process.restart_canary(
                        runtime_root=runtime_root,
                        profile=profile,
                        force=False,
                        stop_timeout=canary_process.DEFAULT_STOP_TIMEOUT,
                        wait_seconds=canary_process.DEFAULT_WAIT_SECONDS,
                        wait_interval=canary_process.DEFAULT_WAIT_INTERVAL,
                    )
                    if result != 0:
                        restart_failed = True
                        decision = "restart_failed"
                    else:
                        state.consecutive_reds = 0
                        state.last_restart_ts = now_ts
            else:
                decision = f"waiting_for_reds={state.consecutive_reds}/{restart_after_reds}"

    status = "GREEN" if is_green else "RED"
    ages = _format_event_ages(rows)
    status_line = (
        f"liveness={status} events={ages} consecutive_reds={state.consecutive_reds} "
        f"decision={decision}"
    )
    return PollOutcome(
        is_green=is_green,
        decision=decision,
        restart_failed=restart_failed,
        status_line=status_line,
    )


def _poll_and_print(
    *,
    runtime_root: Path,
    profile: str,
    event_types: Iterable[str],
    max_age_seconds: int,
    auto_restart: bool,
    restart_after_reds: int,
    restart_cooldown_seconds: int,
    state: WatchdogState,
) -> tuple[bool, bool]:
    try:
        outcome = _poll_once(
            runtime_root=runtime_root,
            profile=profile,
            event_types=event_types,
            max_age_seconds=max_age_seconds,
            auto_restart=auto_restart,
            restart_after_reds=restart_after_reds,
            restart_cooldown_seconds=restart_cooldown_seconds,
            state=state,
        )
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"liveness=ERROR error={exc}")
        return False, True

    print(outcome.status_line)
    return outcome.is_green, outcome.restart_failed


def main() -> int:
    args = _parse_args()
    event_types = args.event_type or list(liveness_check.DEFAULT_EVENT_TYPES)
    runtime_root = args.runtime_root.resolve()
    state_path = _state_path(runtime_root, args.profile)
    state = _load_state(state_path)

    if args.once:
        is_green, failed = _poll_and_print(
            runtime_root=runtime_root,
            profile=args.profile,
            event_types=event_types,
            max_age_seconds=args.max_age_seconds,
            auto_restart=args.auto_restart,
            restart_after_reds=args.restart_after_reds,
            restart_cooldown_seconds=args.restart_cooldown_seconds,
            state=state,
        )
        _save_state(state_path, state)
        if failed:
            return 2
        return 0 if is_green else 1

    while True:
        _is_green, failed = _poll_and_print(
            runtime_root=runtime_root,
            profile=args.profile,
            event_types=event_types,
            max_age_seconds=args.max_age_seconds,
            auto_restart=args.auto_restart,
            restart_after_reds=args.restart_after_reds,
            restart_cooldown_seconds=args.restart_cooldown_seconds,
            state=state,
        )
        _save_state(state_path, state)
        if failed:
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
