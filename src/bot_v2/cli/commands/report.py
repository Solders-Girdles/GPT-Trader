"""Daily report CLI commands."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from bot_v2.cli import options
from bot_v2.monitoring.daily_report import DailyReportGenerator


def register(subparsers: Any) -> None:
    """Register report commands."""
    parser: ArgumentParser = subparsers.add_parser(
        "report", help="Generate performance reports"
    )
    options.add_profile_option(parser)

    report_subparsers = parser.add_subparsers(dest="report_command", required=True)

    # Daily report command
    daily = report_subparsers.add_parser("daily", help="Generate daily performance report")
    options.add_profile_option(daily)
    daily.add_argument(
        "--date",
        type=str,
        help="Date to generate report for (YYYY-MM-DD, defaults to today)",
    )
    daily.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Hours to look back for data (default: 24)",
    )
    daily.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for reports (defaults to {profile}/reports)",
    )
    daily.add_argument(
        "--format",
        choices=["text", "json", "both"],
        default="text",
        help="Output format (default: text)",
    )
    daily.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report to file, just print to stdout",
    )
    daily.set_defaults(handler=_handle_daily_report)


def _handle_daily_report(args: Namespace) -> int:
    """Handle daily report generation."""
    profile = getattr(args, "profile", "demo")

    # Parse date if provided
    report_date = None
    if args.date:
        try:
            report_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD")
            return 1

    # Initialize generator
    generator = DailyReportGenerator(profile=profile)

    # Generate report
    try:
        print(f"Generating daily report for {profile}...")
        report = generator.generate(
            date=report_date, lookback_hours=args.lookback_hours
        )

        # Output based on format
        if args.format in ("text", "both"):
            text = report.to_text()
            if args.no_save:
                print(text)
            else:
                output_dir = Path(args.output_dir) if args.output_dir else None
                path = generator.save_report(report, output_dir=output_dir)
                print(f"\nReport saved to: {path}")
                print("\n" + text)

        if args.format in ("json", "both"):
            report_json = json.dumps(report.to_dict(), indent=2)
            if args.no_save:
                print(report_json)
            elif args.format == "json":
                # Save JSON only
                output_dir = Path(args.output_dir) if args.output_dir else None
                if output_dir is None:
                    output_dir = generator.data_dir / "reports"
                output_dir.mkdir(parents=True, exist_ok=True)
                json_path = output_dir / f"daily_report_{report.date}.json"
                with open(json_path, "w") as f:
                    f.write(report_json)
                print(f"JSON report saved to: {json_path}")

        return 0

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback

        traceback.print_exc()
        return 1
