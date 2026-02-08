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
            generated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
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
    data["timestamp"] = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
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
        "dated 2026-02-01 is 7.0 day(s) old (max 1 day(s))." in output
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
        "dated 2026-02-01 is 7.0 day(s) old (max 1 day(s))." in error_output
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
