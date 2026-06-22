from __future__ import annotations

import argparse
from argparse import Namespace

import gpt_trader.cli.commands.report as report_cmd
from gpt_trader.cli.response import CliResponse


def test_report_daily_profile_argument_inherits_from_parent() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_cmd.register(subparsers)

    from_parent = parser.parse_args(["report", "--profile", "prod", "daily"])
    assert from_parent.profile == "prod"

    from_leaf = parser.parse_args(["report", "daily", "--profile", "prod"])
    assert from_leaf.profile == "prod"


def test_report_daily_json_no_save_returns_report_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubReport:
        date = "2026-06-21"

        def to_dict(self) -> dict[str, object]:
            return {"date": self.date, "total_pnl": 42}

        def to_text(self) -> str:
            return "report text"

    class StubGenerator:
        def __init__(self, *, profile: str):
            captured["profile"] = profile

        def generate(self, *, date, lookback_hours: int):
            captured["date"] = date
            captured["lookback_hours"] = lookback_hours
            return StubReport()

        def save_report(self, report, *, output_dir=None):  # pragma: no cover - not used
            raise AssertionError("no-save JSON mode should not write report files")

    monkeypatch.setattr(report_cmd, "DailyReportGenerator", StubGenerator)

    response = report_cmd._handle_daily_report(
        Namespace(
            output_format="json",
            report_format="text",
            profile="prod",
            date=None,
            lookback_hours=12,
            output_dir=None,
            no_save=True,
        )
    )

    assert isinstance(response, CliResponse)
    assert response.success
    assert captured == {"profile": "prod", "date": None, "lookback_hours": 12}
    assert response.data == {
        "report": {"date": "2026-06-21", "total_pnl": 42},
        "saved_paths": [],
        "profile": "prod",
        "date": "2026-06-21",
    }
