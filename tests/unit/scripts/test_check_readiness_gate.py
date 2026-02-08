from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import scripts.ci.check_readiness_gate as check_readiness_gate

os.environ.setdefault("GPT_TRADER_READINESS_MAX_REPORT_AGE_DAYS", "0")


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "readiness"


def _load_fixture(name: str) -> dict[str, object]:
    path = _fixture_dir() / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_daily_report(
    base_dir: Path,
    report_date: date,
    *,
    profile: str,
    fixture_name: str,
    generated_at: datetime | None = None,
) -> Path:
    data = _load_fixture(fixture_name)
    data["date"] = report_date.isoformat()
    data["profile"] = profile
    if generated_at is None:
        data["generated_at"] = f"{report_date.isoformat()}T00:10:00Z"
    else:
        data["generated_at"] = (
            generated_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    report_dir = base_dir / "runtime_data" / profile / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_report_{report_date.isoformat()}.json"
    report_path.write_text(json.dumps(data), encoding="utf-8")
    return report_path


def _write_preflight_report(
    base_dir: Path,
    report_date: date,
    *,
    profile: str,
    fixture_name: str,
    timestamp: datetime | None = None,
) -> Path:
    data = _load_fixture(fixture_name)
    data["profile"] = profile
    if timestamp is None:
        timestamp = datetime.combine(report_date, time(1, 0), tzinfo=timezone.utc)
    data["timestamp"] = (
        timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    report_path = base_dir / f"preflight_report_{report_date.strftime('%Y%m%d')}_010000.json"
    report_path.write_text(json.dumps(data), encoding="utf-8")
    return report_path


def _freeze_reference_time(monkeypatch, reference_time: datetime) -> None:
    class FrozenDateTime(datetime):
        _reference = reference_time

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return cls._reference
            return cls._reference.astimezone(tz)

    monkeypatch.setattr(check_readiness_gate, "datetime", FrozenDateTime)


def _setup_green_streak(base_dir: Path, profile: str, end_date: date) -> None:
    for offset in range(2, -1, -1):
        report_date = end_date - timedelta(days=offset)
        _write_daily_report(
            base_dir,
            report_date,
            profile=profile,
            fixture_name="daily_report_green.json",
        )
        _write_preflight_report(
            base_dir,
            report_date,
            profile=profile,
            fixture_name="preflight_ready.json",
        )


def test_evaluate_liveness_snapshot_reports_fallback_reason() -> None:
    fallback_payload = {
        "status": "UNKNOWN",
        "max_age_seconds": 300,
        "event_types": ["heartbeat", "price_tick"],
        "events": {
            "heartbeat": {"event_id": None, "timestamp": None, "age_seconds": None},
            "price_tick": {"event_id": None, "timestamp": None, "age_seconds": None},
        },
        "fallback": {"reason": "events.db unavailable", "source": "/runtime_data/canary/events.db"},
    }
    ok, message = check_readiness_gate._evaluate_liveness_snapshot(
        fallback_payload,
        300,
    )
    assert ok is False
    assert "liveness fallback" in message
    assert "events.db unavailable" in message
    assert "source=/runtime_data/canary/events.db" in message


def test_main_passes_with_green_streak(tmp_path: Path, capsys) -> None:
    profile = "canary"
    _setup_green_streak(tmp_path, profile, date(2026, 1, 17))

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Readiness gate PASSED" in output


def test_main_fails_when_preflight_missing(tmp_path: Path, capsys) -> None:
    profile = "canary"
    end_date = date(2026, 1, 17)
    for offset in range(2, -1, -1):
        report_date = end_date - timedelta(days=offset)
        _write_daily_report(
            tmp_path,
            report_date,
            profile=profile,
            fixture_name="daily_report_green.json",
        )
        if offset != 1:
            _write_preflight_report(
                tmp_path,
                report_date,
                profile=profile,
                fixture_name="preflight_ready.json",
            )

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
        ]
    )

    error_output = capsys.readouterr().err
    assert result == 1
    assert "missing preflight report" in error_output


def test_main_fails_when_pillar_not_green(tmp_path: Path, capsys) -> None:
    profile = "canary"
    end_date = date(2026, 1, 17)
    _setup_green_streak(tmp_path, profile, end_date)
    _write_daily_report(
        tmp_path,
        end_date,
        profile=profile,
        fixture_name="daily_report_bad.json",
    )

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
        ]
    )

    error_output = capsys.readouterr().err
    assert result == 1
    assert "Execution correctness" in error_output
    assert "api_errors" in error_output


def test_main_skips_when_no_reports(tmp_path: Path, capsys) -> None:
    result = check_readiness_gate.main(
        [
            "--profile",
            "canary",
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Readiness gate skipped" in output


def test_main_require_reports_fails_without_reports(tmp_path: Path, capsys) -> None:
    result = check_readiness_gate.main(
        [
            "--profile",
            "canary",
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
            "--require-reports",
        ]
    )

    error_output = capsys.readouterr().err
    assert result == 1
    assert "Readiness gate skipped" in error_output


def test_main_scopes_daily_report_parsing_to_target_profile(tmp_path: Path, capsys) -> None:
    """A broken report for another profile should not fail the canary gate."""

    profile = "canary"
    _setup_green_streak(tmp_path, profile, date(2026, 1, 17))

    other_profile_report = (
        tmp_path / "runtime_data" / "dev" / "reports" / "daily_report_2026-01-17.json"
    )
    other_profile_report.parent.mkdir(parents=True, exist_ok=True)
    other_profile_report.write_text("{not valid json", encoding="utf-8")

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Readiness gate PASSED" in output


def test_main_degrades_when_reports_are_stale(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    profile = "canary"
    reference_time = datetime(2026, 2, 8, tzinfo=timezone.utc)
    _freeze_reference_time(monkeypatch, reference_time)
    stale_date = date(2026, 2, 1)
    _write_daily_report(
        base_dir=tmp_path,
        report_date=stale_date,
        profile=profile,
        fixture_name="daily_report_green.json",
        generated_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    _write_preflight_report(
        tmp_path,
        stale_date,
        profile=profile,
        fixture_name="preflight_ready.json",
        timestamp=reference_time,
    )

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
            "--max-report-age-days",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert (
        "Readiness gate degraded: latest daily report for profile 'canary' "
        "dated 2026-02-01 is 7.0 day(s) old (max 1 day(s))."
        in output
    )
    assert "Set --strict" in output
    assert "Readiness gate PASSED" not in output
    assert "Readiness gate FAILED" not in output
    assert "GREEN streak" not in output


def test_main_strict_mode_fails_when_reports_are_stale(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    profile = "canary"
    reference_time = datetime(2026, 2, 8, tzinfo=timezone.utc)
    _freeze_reference_time(monkeypatch, reference_time)
    stale_date = date(2026, 2, 1)
    _write_daily_report(
        base_dir=tmp_path,
        report_date=stale_date,
        profile=profile,
        fixture_name="daily_report_green.json",
        generated_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    _write_preflight_report(
        tmp_path,
        stale_date,
        profile=profile,
        fixture_name="preflight_ready.json",
        timestamp=reference_time,
    )

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
            "--max-report-age-days",
            "1",
            "--strict",
        ]
    )

    error_output = capsys.readouterr().err
    assert result == 1
    assert (
        "Readiness gate degraded: latest daily report for profile 'canary' "
        "dated 2026-02-01 is 7.0 day(s) old (max 1 day(s))."
        in error_output
    )
    assert "Readiness gate FAILED: stale daily report exceeds max age." in error_output
    assert "GREEN streak" not in error_output


def test_main_passes_when_latest_report_age_matches_max_age(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    profile = "canary"
    reference_time = datetime(2026, 2, 8, tzinfo=timezone.utc)
    _freeze_reference_time(monkeypatch, reference_time)
    latest_date = date(2026, 2, 3)
    for offset in range(2, -1, -1):
        report_date = latest_date - timedelta(days=offset)
        generated_at = None
        if report_date == latest_date:
            generated_at = datetime(
                report_date.year,
                report_date.month,
                report_date.day,
                tzinfo=timezone.utc,
            )
        _write_daily_report(
            base_dir=tmp_path,
            report_date=report_date,
            profile=profile,
            fixture_name="daily_report_green.json",
            generated_at=generated_at,
        )
        preflight_timestamp = datetime(
            report_date.year,
            report_date.month,
            report_date.day,
            1,
            0,
            tzinfo=timezone.utc,
        )
        _write_preflight_report(
            tmp_path,
            report_date,
            profile=profile,
            fixture_name="preflight_ready.json",
            timestamp=preflight_timestamp,
        )

    result = check_readiness_gate.main(
        [
            "--profile",
            profile,
            "--daily-root",
            str(tmp_path / "runtime_data"),
            "--preflight-dir",
            str(tmp_path),
            "--max-report-age-days",
            "5",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Readiness gate PASSED" in output
    assert "Readiness gate degraded" not in output


def test_find_latest_report_for_profile_prefers_newest_report_date(tmp_path: Path) -> None:
    profile = "canary"
    report_dir = tmp_path / "runtime_data" / profile / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    older_path = report_dir / "daily_report_2026-01-15.json"
    newer_path = report_dir / "daily_report_2026-01-16.json"
    older_path.write_text("{}", encoding="utf-8")
    newer_path.write_text("{}", encoding="utf-8")

    entries = [
        check_readiness_gate.ReportEntry(
            report_date=date(2026, 1, 15),
            profile=profile,
            path=older_path,
            generated_at=datetime(2026, 1, 15, 23, 59, tzinfo=timezone.utc),
            data={},
        ),
        check_readiness_gate.ReportEntry(
            report_date=date(2026, 1, 16),
            profile=profile,
            path=newer_path,
            generated_at=None,
            data={},
        ),
    ]

    latest = check_readiness_gate._find_latest_report_for_profile(entries, profile)
    assert latest is not None
    assert latest.report_date == date(2026, 1, 16)
    assert latest.path == newer_path
