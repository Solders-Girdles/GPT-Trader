#!/usr/bin/env python3
"""Readiness gate automation based on daily + preflight JSON reports.

Usage:
    python scripts/ci/check_readiness_gate.py --profile canary
    python scripts/ci/check_readiness_gate.py --profile canary --update-docs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_START = "<!-- readiness-streak-table:start -->"
TABLE_END = "<!-- readiness-streak-table:end -->"


@dataclass(frozen=True)
class Thresholds:
    stale_marks_max: int
    ws_reconnects_max: int
    unfilled_orders_max: int
    api_errors_max: int
    guard_triggers_max: int
    liveness_max_age_seconds: int


@dataclass(frozen=True)
class ReportEntry:
    report_date: date
    profile: str
    path: Path
    generated_at: datetime | None
    data: dict[str, Any]


@dataclass(frozen=True)
class PillarEvaluation:
    name: str
    green: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DayEvaluation:
    report_date: date
    daily_path: Path | None
    preflight_path: Path | None
    green: bool
    notes: tuple[str, ...]
    pillars: tuple[PillarEvaluation, ...]
    preflight_status: str | None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate readiness pillars from daily/preflight reports and enforce a "
            "multi-day GREEN streak."
        )
    )
    parser.add_argument(
        "--profile",
        default="canary",
        help="Profile to evaluate (default: canary).",
    )
    parser.add_argument(
        "--daily-root",
        default=str(REPO_ROOT / "runtime_data"),
        help="Root directory containing runtime_data/<profile>/reports.",
    )
    parser.add_argument(
        "--preflight-dir",
        default=str(REPO_ROOT),
        help="Directory containing preflight_report_*.json (default: repo root).",
    )
    parser.add_argument(
        "--streak-days",
        type=int,
        default=3,
        help="Number of consecutive GREEN days required (default: 3).",
    )
    parser.add_argument(
        "--require-reports",
        action="store_true",
        help="Fail if no daily reports are found (default: skip).",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help="Update docs/READINESS.md with the generated streak table.",
    )
    parser.add_argument(
        "--docs-path",
        default=str(REPO_ROOT / "docs" / "READINESS.md"),
        help="Path to READINESS.md (used with --update-docs).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output instead of the human-readable table.",
    )
    return parser.parse_args(argv)


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_report_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    value = text.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_daily_report_date(path: Path, data: dict[str, Any]) -> date | None:
    date_value = data.get("date")
    report_date = _parse_report_date(str(date_value)) if date_value is not None else None
    if report_date is not None:
        return report_date
    match = re.search(r"daily_report_(\d{4}-\d{2}-\d{2})\.json$", path.name)
    if match:
        return _parse_report_date(match.group(1))
    return None


def _parse_preflight_report_date(path: Path, data: dict[str, Any]) -> date | None:
    timestamp_value = data.get("timestamp")
    timestamp = _parse_timestamp(str(timestamp_value)) if timestamp_value else None
    if timestamp is not None:
        return timestamp.date()
    match = re.search(r"preflight_report_(\d{8})_(\d{6})\.json$", path.name)
    if match:
        try:
            parsed = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc).date()
    return None


def _infer_profile_from_path(path: Path) -> str | None:
    parts = list(path.parts)
    if "runtime_data" not in parts:
        return None
    index = parts.index("runtime_data")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return None, f"Failed to read {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"Report is not a JSON object: {path}"
    return data, None


def _discover_daily_reports(root: Path) -> tuple[list[ReportEntry], list[str]]:
    errors: list[str] = []
    reports: list[ReportEntry] = []
    if not root.exists():
        return reports, errors
    for path in sorted(root.rglob("daily_report_*.json")):
        data, error = _load_json(path)
        if error:
            errors.append(error)
            continue
        assert data is not None
        report_date = _parse_daily_report_date(path, data)
        if report_date is None:
            errors.append(f"Could not determine date for daily report: {path}")
            continue
        profile = data.get("profile")
        if not profile:
            inferred = _infer_profile_from_path(path)
            profile = inferred or "unknown"
        generated_at = _parse_timestamp(str(data.get("generated_at")))
        reports.append(
            ReportEntry(
                report_date=report_date,
                profile=str(profile),
                path=path,
                generated_at=generated_at,
                data=data,
            )
        )
    return reports, errors


def _discover_preflight_reports(root: Path) -> tuple[list[ReportEntry], list[str]]:
    errors: list[str] = []
    reports: list[ReportEntry] = []
    if not root.exists():
        return reports, errors
    for path in sorted(root.glob("preflight_report_*.json")):
        data, error = _load_json(path)
        if error:
            errors.append(error)
            continue
        assert data is not None
        report_date = _parse_preflight_report_date(path, data)
        if report_date is None:
            errors.append(f"Could not determine date for preflight report: {path}")
            continue
        profile = data.get("profile", "unknown")
        generated_at = _parse_timestamp(str(data.get("timestamp")))
        reports.append(
            ReportEntry(
                report_date=report_date,
                profile=str(profile),
                path=path,
                generated_at=generated_at,
                data=data,
            )
        )
    return reports, errors


def _select_latest(entries: list[ReportEntry]) -> ReportEntry:
    def sort_key(entry: ReportEntry) -> tuple[int, float]:
        if entry.generated_at is not None:
            return (1, entry.generated_at.timestamp())
        try:
            return (0, entry.path.stat().st_mtime)
        except OSError:
            return (0, 0.0)

    return max(entries, key=sort_key)


def _evaluate_liveness_snapshot(
    liveness: dict[str, Any] | None,
    max_age_seconds: int,
) -> tuple[bool, str | None]:
    if not isinstance(liveness, dict):
        return False, "missing liveness snapshot"
    events = liveness.get("events")
    if not isinstance(events, dict):
        return False, "missing liveness events"
    ages: dict[str, float] = {}
    for event_type, details in events.items():
        if not isinstance(details, dict):
            continue
        age_value = details.get("age_seconds")
        if isinstance(age_value, (int, float)):
            ages[str(event_type)] = float(age_value)
    if not ages:
        return False, "missing liveness ages"
    best_event = min(ages, key=ages.get)
    best_age = ages[best_event]
    if best_age <= max_age_seconds:
        return True, None
    return False, f"liveness {best_event}: {int(best_age)}s > {max_age_seconds}s"


def _evaluate_market_data(
    health: dict[str, Any] | None,
    thresholds: Thresholds,
) -> PillarEvaluation:
    reasons: list[str] = []
    if not isinstance(health, dict):
        return PillarEvaluation(
            name="Market data integrity",
            green=False,
            reasons=("missing health section",),
        )

    stale_marks = _coerce_int(health.get("stale_marks", health.get("stale_marks_count")))
    ws_reconnects = _coerce_int(health.get("ws_reconnects"))
    if stale_marks > thresholds.stale_marks_max:
        reasons.append(f"stale_marks {stale_marks} > {thresholds.stale_marks_max}")
    if ws_reconnects > thresholds.ws_reconnects_max:
        reasons.append(f"ws_reconnects {ws_reconnects} > {thresholds.ws_reconnects_max}")

    liveness_ok, liveness_reason = _evaluate_liveness_snapshot(
        health.get("liveness"),
        thresholds.liveness_max_age_seconds,
    )
    if not liveness_ok and liveness_reason:
        reasons.append(liveness_reason)

    return PillarEvaluation(
        name="Market data integrity",
        green=not reasons,
        reasons=tuple(reasons),
    )


def _evaluate_execution(
    health: dict[str, Any] | None,
    thresholds: Thresholds,
) -> PillarEvaluation:
    reasons: list[str] = []
    if not isinstance(health, dict):
        return PillarEvaluation(
            name="Execution correctness",
            green=False,
            reasons=("missing health section",),
        )

    unfilled_orders = _coerce_int(health.get("unfilled_orders"))
    api_errors = _coerce_int(health.get("api_errors"))
    if unfilled_orders > thresholds.unfilled_orders_max:
        reasons.append(f"unfilled_orders {unfilled_orders} > {thresholds.unfilled_orders_max}")
    if api_errors > thresholds.api_errors_max:
        reasons.append(f"api_errors {api_errors} > {thresholds.api_errors_max}")

    return PillarEvaluation(
        name="Execution correctness",
        green=not reasons,
        reasons=tuple(reasons),
    )


def _evaluate_risk(
    risk: dict[str, Any] | None,
    thresholds: Thresholds,
) -> PillarEvaluation:
    if not isinstance(risk, dict):
        return PillarEvaluation(
            name="Risk management",
            green=False,
            reasons=("missing risk section",),
        )
    reasons: list[str] = []

    guard_triggers = risk.get("guard_triggers", {})
    guard_total = 0
    if isinstance(guard_triggers, dict):
        guard_total = sum(_coerce_int(value) for value in guard_triggers.values())
    else:
        guard_total = _coerce_int(guard_triggers)
    if guard_total > thresholds.guard_triggers_max:
        reasons.append(f"guard_triggers {guard_total} > {thresholds.guard_triggers_max}")

    circuit_breaker = risk.get("circuit_breaker_state", {})
    if isinstance(circuit_breaker, dict) and circuit_breaker.get("triggered"):
        reasons.append("circuit_breaker triggered")

    return PillarEvaluation(
        name="Risk management",
        green=not reasons,
        reasons=tuple(reasons),
    )


def _evaluate_daily_report(
    report: dict[str, Any],
    thresholds: Thresholds,
) -> tuple[tuple[PillarEvaluation, ...], list[str]]:
    health = report.get("health")
    risk = report.get("risk")
    pillars = (
        _evaluate_market_data(health, thresholds),
        _evaluate_risk(risk, thresholds),
        _evaluate_execution(health, thresholds),
    )
    notes: list[str] = []
    for pillar in pillars:
        if pillar.green:
            continue
        for reason in pillar.reasons:
            notes.append(f"{pillar.name}: {reason}")
    return pillars, notes


def _build_lookup(
    reports: list[ReportEntry],
    profile: str,
) -> dict[date, ReportEntry]:
    by_date: dict[date, list[ReportEntry]] = {}
    for report in reports:
        if report.profile != profile:
            continue
        by_date.setdefault(report.report_date, []).append(report)
    return {report_date: _select_latest(entries) for report_date, entries in by_date.items()}


def _evaluate_profile(
    profile: str,
    daily_reports: list[ReportEntry],
    preflight_reports: list[ReportEntry],
    thresholds: Thresholds,
    streak_days: int,
) -> tuple[list[DayEvaluation], bool]:
    daily_by_date = _build_lookup(daily_reports, profile)
    preflight_by_date = _build_lookup(preflight_reports, profile)

    if not daily_by_date:
        return [], False

    latest_date = max(daily_by_date)
    required_dates = [
        latest_date - timedelta(days=offset) for offset in range(streak_days - 1, -1, -1)
    ]
    evaluations: list[DayEvaluation] = []

    for report_date in required_dates:
        daily_entry = daily_by_date.get(report_date)
        preflight_entry = preflight_by_date.get(report_date)
        notes: list[str] = []
        pillars: tuple[PillarEvaluation, ...] = ()
        green = True

        if daily_entry is None:
            green = False
            notes.append("missing daily report")
        else:
            pillars, pillar_notes = _evaluate_daily_report(daily_entry.data, thresholds)
            if pillar_notes:
                green = False
                notes.extend(pillar_notes)

        preflight_status: str | None = None
        if preflight_entry is None:
            green = False
            notes.append("missing preflight report")
        else:
            preflight_status = str(preflight_entry.data.get("status", "UNKNOWN"))
            if preflight_status != "READY":
                green = False
                notes.append(f"preflight status {preflight_status}")

        evaluations.append(
            DayEvaluation(
                report_date=report_date,
                daily_path=daily_entry.path if daily_entry else None,
                preflight_path=preflight_entry.path if preflight_entry else None,
                green=green,
                notes=tuple(notes),
                pillars=pillars,
                preflight_status=preflight_status,
            )
        )

    streak_green = all(entry.green for entry in evaluations)
    return evaluations, streak_green


def _format_table(entries: list[DayEvaluation], root: Path) -> str:
    lines = [
        "| Date (UTC) | Daily report path | Preflight path | Green? | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        daily_path = "N/A"
        preflight_path = "N/A"
        if entry.daily_path is not None:
            resolved = entry.daily_path.resolve()
            try:
                resolved = resolved.relative_to(root)
            except ValueError:
                pass
            daily_path = f"`{resolved}`"
        if entry.preflight_path is not None:
            resolved = entry.preflight_path.resolve()
            try:
                resolved = resolved.relative_to(root)
            except ValueError:
                pass
            preflight_path = f"`{resolved}`"
        green_label = "Yes" if entry.green else "No"
        notes = "; ".join(entry.notes) if entry.notes else "OK"
        lines.append(
            f"| {entry.report_date.isoformat()} | {daily_path} | {preflight_path} | {green_label} | {notes} |"
        )
    return "\n".join(lines)


def _relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        relative = resolved
    return str(relative)


def _day_payload(entry: DayEvaluation) -> dict[str, Any]:
    return {
        "date": entry.report_date.isoformat(),
        "green": entry.green,
        "notes": list(entry.notes),
        "pillars": [
            {"name": pillar.name, "green": pillar.green, "reasons": list(pillar.reasons)}
            for pillar in entry.pillars
        ],
        "daily_report_path": _relative_path(entry.daily_path),
        "preflight_path": _relative_path(entry.preflight_path),
        "preflight_status": entry.preflight_status,
    }


def _collect_failure_reasons(entries: list[DayEvaluation]) -> list[str]:
    reasons: list[str] = []
    for entry in entries:
        if entry.green:
            continue
        entry_notes = "; ".join(entry.notes) if entry.notes else "not green"
        reasons.append(f"{entry.report_date.isoformat()}: {entry_notes}")
    return reasons


def _build_json_payload(
    profile: str,
    streak_days: int,
    thresholds: Thresholds,
    evaluations: list[DayEvaluation],
    streak_green: bool,
    status: str,
    status_message: str,
    extra_failure_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    dates = [entry.report_date.isoformat() for entry in evaluations]
    streak_window = {
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "dates": dates,
    }
    failure_reasons = list(_collect_failure_reasons(evaluations))
    if extra_failure_reasons:
        failure_reasons.extend(extra_failure_reasons)
    return {
        "profile": profile,
        "streak_days": streak_days,
        "streak_window": streak_window,
        "streak_green": streak_green,
        "status": status,
        "status_message": status_message,
        "thresholds": {
            "stale_marks_max": thresholds.stale_marks_max,
            "ws_reconnects_max": thresholds.ws_reconnects_max,
            "unfilled_orders_max": thresholds.unfilled_orders_max,
            "api_errors_max": thresholds.api_errors_max,
            "guard_triggers_max": thresholds.guard_triggers_max,
            "liveness_max_age_seconds": thresholds.liveness_max_age_seconds,
        },
        "days": [_day_payload(entry) for entry in evaluations],
        "failure_reasons": failure_reasons,
    }


def _update_docs_table(docs_path: Path, table_text: str) -> None:
    content = docs_path.read_text(encoding="utf-8")
    start_index = content.find(TABLE_START)
    end_index = content.find(TABLE_END)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise RuntimeError("READINESS.md table markers not found")
    start_index += len(TABLE_START)
    new_section = f"\n{table_text}\n"
    updated = content[:start_index] + new_section + content[end_index:]
    docs_path.write_text(updated, encoding="utf-8")


def _thresholds_from_env() -> Thresholds:
    return Thresholds(
        stale_marks_max=_get_env_int("GPT_TRADER_READINESS_STALE_MARKS_MAX", 0),
        ws_reconnects_max=_get_env_int("GPT_TRADER_READINESS_WS_RECONNECTS_MAX", 3),
        unfilled_orders_max=_get_env_int("GPT_TRADER_READINESS_UNFILLED_ORDERS_MAX", 0),
        api_errors_max=_get_env_int("GPT_TRADER_READINESS_API_ERRORS_MAX", 0),
        guard_triggers_max=_get_env_int("GPT_TRADER_READINESS_GUARD_TRIGGERS_MAX", 0),
        liveness_max_age_seconds=_get_env_int(
            "GPT_TRADER_READINESS_LIVENESS_MAX_AGE_SECONDS",
            300,
        ),
    )


def _resolve_daily_reports_root(daily_root: Path, profile: str) -> Path:
    """Resolve the directory to scan for daily reports.

    Historically we pointed --daily-root at runtime_data and then rglobbed for
    daily_report_*.json across *all* profiles.

    This resolver scopes the search to the requested profile (preferred) while
    remaining backwards compatible if a caller already points at
    runtime_data/<profile> or runtime_data/<profile>/reports.
    """

    if daily_root.name == "reports":
        return daily_root

    candidate = daily_root / "reports"
    if candidate.exists():
        return candidate

    return daily_root / profile / "reports"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = str(args.profile)
    streak_days = int(args.streak_days)
    daily_root = Path(args.daily_root).expanduser()
    preflight_dir = Path(args.preflight_dir).expanduser()

    daily_reports_root = _resolve_daily_reports_root(daily_root, profile)

    daily_reports, daily_errors = _discover_daily_reports(daily_reports_root)
    preflight_reports, preflight_errors = _discover_preflight_reports(preflight_dir)
    errors = daily_errors + preflight_errors

    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    thresholds = _thresholds_from_env()
    evaluations, streak_green = _evaluate_profile(
        profile,
        daily_reports,
        preflight_reports,
        thresholds,
        streak_days,
    )

    if not evaluations:
        message = (
            f"Readiness gate skipped: no daily reports found for profile '{profile}' "
            f"under {daily_root}."
        )
        if args.require_reports:
            if args.json:
                payload = _build_json_payload(
                    profile=profile,
                    streak_days=streak_days,
                    thresholds=thresholds,
                    evaluations=evaluations,
                    streak_green=streak_green,
                    status="FAILED",
                    status_message=message,
                    extra_failure_reasons=(message,),
                )
                print(json.dumps(payload))
                return 1
            print(f"Error: {message}", file=sys.stderr)
            return 1
        if args.json:
            payload = _build_json_payload(
                profile=profile,
                streak_days=streak_days,
                thresholds=thresholds,
                evaluations=evaluations,
                streak_green=streak_green,
                status="SKIPPED",
                status_message=message,
                extra_failure_reasons=(message,),
            )
            print(json.dumps(payload))
            return 0
        print(message)
        return 0

    table_text = _format_table(evaluations, REPO_ROOT)

    if not args.json:
        print(
            "Readiness gate (profile={profile}, streak_days={streak_days})".format(
                profile=profile,
                streak_days=streak_days,
            )
        )
        print(
            "Thresholds: stale_marks<={stale_marks}, ws_reconnects<={ws_reconnects}, "
            "unfilled_orders<={unfilled}, api_errors<={api_errors}, guard_triggers<={guard}, "
            "liveness_max_age_seconds<={liveness}".format(
                stale_marks=thresholds.stale_marks_max,
                ws_reconnects=thresholds.ws_reconnects_max,
                unfilled=thresholds.unfilled_orders_max,
                api_errors=thresholds.api_errors_max,
                guard=thresholds.guard_triggers_max,
                liveness=thresholds.liveness_max_age_seconds,
            )
        )
        print("\n" + table_text + "\n")

    if args.update_docs:
        docs_path = Path(args.docs_path).expanduser()
        _update_docs_table(docs_path, table_text)
        if args.json:
            print(f"Updated {docs_path}", file=sys.stderr)
        else:
            print(f"Updated {docs_path}")

    status = "PASSED" if streak_green else "FAILED"
    status_message = (
        f"Readiness gate PASSED: {streak_days}-day GREEN streak satisfied."
        if streak_green
        else f"Readiness gate FAILED: {streak_days}-day GREEN streak not satisfied."
    )
    payload = _build_json_payload(
        profile=profile,
        streak_days=streak_days,
        thresholds=thresholds,
        evaluations=evaluations,
        streak_green=streak_green,
        status=status,
        status_message=status_message,
    )
    if args.json:
        print(json.dumps(payload))

    if not streak_green:
        print(status_message, file=sys.stderr)
        for entry in evaluations:
            if entry.green:
                continue
            reasons = "; ".join(entry.notes) if entry.notes else "not green"
            print(f"- {entry.report_date.isoformat()}: {reasons}", file=sys.stderr)
        return 1

    if not args.json:
        print(status_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
